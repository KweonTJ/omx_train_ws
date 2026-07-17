"""Warm-start the PPO actor from collision-free reference trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from stable_baselines3 import PPO
import torch
from torch.nn import functional as torch_functional

from envs import load_config, make_grasp_env
from envs.factory import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]


def solve_pregrasp_joints(env, object_position: np.ndarray) -> np.ndarray:
    target = env._pregrasp_target(object_position)
    joint1 = env._bearing(target)
    scratch = mujoco.MjData(env.model)
    mujoco.mj_resetDataKeyframe(env.model, scratch, env._stay_key_id)

    def residual(joints_2_to_4: np.ndarray) -> np.ndarray:
        joints = np.concatenate(([joint1], joints_2_to_4))
        scratch.qpos[env._arm_qpos_addr] = joints
        scratch.ctrl[:4] = joints
        mujoco.mj_forward(env.model, scratch)
        position_error = 10.0 * (
            scratch.site_xpos[env._eef_site_id] - target
        )
        return np.concatenate((position_error, [np.sum(joints_2_to_4)]))

    solution = least_squares(
        residual,
        x0=np.array([1.15, -0.48, -0.67]),
        bounds=(env._ctrl_low[1:], env._ctrl_high[1:]),
    )
    if not solution.success:
        raise RuntimeError(f"Reference IK failed: {solution.message}")
    return np.concatenate(([joint1], solution.x))


def reference_action(env, info: dict[str, Any], pregrasp: np.ndarray) -> np.ndarray:
    if info["is_grasped"]:
        joint_goal = env._stay_joint_positions
    elif info["approach_stage"] < info["approach_stage_count"]:
        joint_goal = env._episode_approach_joint_waypoints[
            info["approach_stage"]
        ]
    else:
        joint_goal = pregrasp
    return np.clip(
        (joint_goal - env._arm_target) / env.action_scale,
        -1.0,
        1.0,
    ).astype(np.float32)


def collect_demonstrations(
    config: dict[str, Any], stage: str, episodes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    env = make_grasp_env(config, stage_name=stage)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    successes = 0

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        pregrasp = solve_pregrasp_joints(env, info["object_initial_position"])
        for _ in range(env.max_episode_steps):
            action = reference_action(env, info, pregrasp)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(info["is_success"])

    env.close()
    if successes != episodes:
        raise RuntimeError(
            f"Reference demonstrations failed: {successes}/{episodes}"
        )
    print(
        f"demonstrations: episodes={episodes}, transitions={len(observations)}, "
        f"success={successes}/{episodes}"
    )
    return np.asarray(observations, dtype=np.float32), np.asarray(
        actions, dtype=np.float32
    )


def collect_dagger_demonstrations(
    model: PPO,
    config: dict[str, Any],
    stage: str,
    episodes: int,
    seed: int,
    expert_blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    env = make_grasp_env(config, stage_name=stage)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    successes = 0
    collisions = 0

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        pregrasp = solve_pregrasp_joints(env, info["object_initial_position"])
        for _ in range(env.max_episode_steps):
            expert_action = reference_action(env, info, pregrasp)
            policy_action, _ = model.predict(observation, deterministic=True)
            observations.append(observation.copy())
            actions.append(expert_action.copy())
            executed_action = np.clip(
                expert_blend * expert_action
                + (1.0 - expert_blend) * policy_action,
                -1.0,
                1.0,
            )
            observation, _, terminated, truncated, info = env.step(
                executed_action
            )
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        collisions += int(info.get("collision_failure", False))

    env.close()
    print(
        f"DAgger data: episodes={episodes}, transitions={len(observations)}, "
        f"expert_blend={expert_blend:.2f}, success={successes}/{episodes}, "
        f"collisions={collisions}/{episodes}"
    )
    return np.asarray(observations, dtype=np.float32), np.asarray(
        actions, dtype=np.float32
    )


def make_model(config: dict[str, Any], stage: str, seed: int) -> PPO:
    ppo = config["ppo"]
    policy = ppo["policy"]
    activations = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}
    env = make_grasp_env(config, stage_name=stage)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        vf_coef=float(ppo["vf_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        target_kl=float(ppo["target_kl"]),
        policy_kwargs={
            "net_arch": list(policy["net_arch"]),
            "activation_fn": activations[policy["activation"].lower()],
        },
        seed=seed,
        device=ppo.get("device", "cpu"),
        verbose=0,
    )


def actor_mean(model: PPO, observations: torch.Tensor) -> torch.Tensor:
    features = model.policy.extract_features(observations)
    latent_policy = model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent_policy)


def pretrain_actor(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    split = max(1, int(0.9 * len(observations)))
    indices = rng.permutation(len(observations))
    train_indices, validation_indices = indices[:split], indices[split:]
    optimizer = torch.optim.Adam(
        model.policy.parameters(), learning_rate, weight_decay=1.0e-6
    )
    device = model.policy.device

    for epoch in range(epochs):
        rng.shuffle(train_indices)
        model.policy.train()
        losses = []
        for start in range(0, len(train_indices), batch_size):
            batch = train_indices[start : start + batch_size]
            observation_tensor = torch.as_tensor(
                observations[batch], device=device
            )
            action_tensor = torch.as_tensor(actions[batch], device=device)
            predicted = actor_mean(model, observation_tensor)
            loss = torch_functional.mse_loss(predicted, action_tensor)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.policy.eval()
        with torch.no_grad():
            validation_observations = torch.as_tensor(
                observations[validation_indices], device=device
            )
            validation_actions = torch.as_tensor(
                actions[validation_indices], device=device
            )
            validation_loss = torch_functional.mse_loss(
                actor_mean(model, validation_observations), validation_actions
            )
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_mse={np.mean(losses):.6f} "
                f"validation_mse={float(validation_loss.cpu()):.6f}"
            )


def evaluate_closed_loop(
    model: PPO,
    config: dict[str, Any],
    stage: str,
    episodes: int,
    seed: int,
) -> tuple[int, int, int]:
    env = make_grasp_env(config, stage_name=stage)
    successes = 0
    grasps = 0
    collisions = 0
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        for _ in range(env.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        grasps += int(info["is_grasped"])
        collisions += int(info.get("collision_failure", False))
    env.close()
    return successes, grasps, collisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default="center_mixed")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=100)
    parser.add_argument("--dagger-epochs", type=int, default=10)
    parser.add_argument(
        "--dagger-stage",
        action="append",
        help="Stage to aggregate each DAgger round. Repeat for balanced stages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "policies/latest/arm_grasp/arm_grasp_bc_warmstart.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("episodes cannot be negative; epochs and batch-size must be positive")
    if args.dagger_rounds < 0 or args.dagger_episodes < 1 or args.dagger_epochs < 1:
        raise ValueError("DAgger settings must be positive, except rounds may be zero")
    if args.episodes == 0 and args.resume is None:
        raise ValueError("--episodes 0 requires --resume")
    config = load_config(args.config)
    torch.set_num_threads(int(config["ppo"].get("torch_threads", 1)))
    if args.resume:
        env = make_grasp_env(config, stage_name=args.stage)
        model = PPO.load(args.resume.expanduser().resolve(), env=env, device="cpu")
    else:
        model = make_model(config, args.stage, args.seed)

    observation_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    if args.episodes:
        observations, actions = collect_demonstrations(
            config, args.stage, args.episodes, args.seed
        )
        observation_batches.append(observations)
        action_batches.append(actions)
        pretrain_actor(
            model,
            observations,
            actions,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
        )

    dagger_stages = args.dagger_stage or [args.stage]
    for round_index in range(args.dagger_rounds):
        if args.dagger_rounds == 1:
            expert_blend = 0.25
        else:
            expert_blend = 0.50 * (
                1.0 - round_index / (args.dagger_rounds - 1)
            )
        for stage_index, dagger_stage in enumerate(dagger_stages):
            observations, actions = collect_dagger_demonstrations(
                model,
                config,
                dagger_stage,
                args.dagger_episodes,
                args.seed
                + 100000 * (round_index + 1)
                + 10000 * stage_index,
                expert_blend,
            )
            observation_batches.append(observations)
            action_batches.append(actions)
        pretrain_actor(
            model,
            np.concatenate(observation_batches),
            np.concatenate(action_batches),
            args.dagger_epochs,
            args.batch_size,
            args.learning_rate,
            args.seed + round_index + 1,
        )

    for stage in ("center_reach", "grasp_return_center", "center_mixed"):
        successes, grasps, collisions = evaluate_closed_loop(
            model, config, stage, 20, args.seed + 10000
        )
        print(
            f"closed_loop stage={stage}: success={successes}/20, "
            f"grasped={grasps}/20, collisions={collisions}/20"
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    model.get_env().close()
    print(f"Saved BC warm-start: {output}")


if __name__ == "__main__":
    main()
