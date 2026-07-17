"""Train the randomized arm grasp policy with a staged PPO curriculum."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import load_config, make_grasp_env
from envs.factory import DEFAULT_CONFIG, environment_kwargs


ROOT = Path(__file__).resolve().parents[1]
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")


def _env_factory(
    config: dict[str, Any], stage_name: str, seed: int
) -> Callable[[], gym.Env]:
    def create() -> gym.Env:
        env = make_grasp_env(config, stage_name=stage_name)
        env.reset(seed=seed)
        return Monitor(
            env,
            info_keywords=(
                "is_success",
                "is_grasped",
                "released",
                "episode_task",
                "object_sample_bucket",
                "collision_failure",
            ),
        )

    return create


def _make_vector_env(
    config: dict[str, Any], stage_name: str, n_envs: int, seed: int
) -> DummyVecEnv:
    return DummyVecEnv(
        [_env_factory(config, stage_name, seed + index) for index in range(n_envs)]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _activation(name: str) -> type[torch.nn.Module]:
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported policy activation: {name}") from error


def _joint_qpos_addresses(
    model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> np.ndarray:
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Missing joint in MuJoCo model: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int32)


def _write_metadata(
    output_dir: Path,
    config: dict[str, Any],
    completed_stages: list[str],
    total_timesteps: int,
) -> None:
    config_path = Path(config["_config_path"])
    model_path = Path(environment_kwargs(config)["model_path"])
    environment_config = config["environment"]
    mujoco_model = mujoco.MjModel.from_xml_path(str(model_path))
    stay_key_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_KEY, "stay"
    )
    if stay_key_id < 0:
        raise ValueError(f"Missing stay keyframe in {model_path}")
    arm_qpos_addresses = _joint_qpos_addresses(
        mujoco_model, ARM_JOINT_NAMES
    )
    metadata = {
        "policy_version": output_dir.name,
        "policy_name": config["experiment"]["name"],
        "framework": "stable-baselines3 PPO",
        "completed_stages": completed_stages,
        "total_timesteps": int(total_timesteps),
        "observation_schema_version": 1,
        "observation_size": int(config["model"]["expected_observation_size"]),
        "action_schema_version": 2,
        "action_size": int(config["model"]["expected_action_size"]),
        "joint_names": list(ARM_JOINT_NAMES),
        "control_mode": "reference_plus_residual",
        "control_period_s": float(
            mujoco_model.opt.timestep * environment_config["frame_skip"]
        ),
        "action_scale": [
            float(value) for value in environment_config["action_scale"]
        ],
        "action_filter_coefficient": float(
            environment_config["action_filter_coefficient"]
        ),
        "residual_action_scale": float(
            environment_config["residual_action_scale"]
        ),
        "stay_joint_positions": [
            float(value)
            for value in mujoco_model.key_qpos[
                stay_key_id, arm_qpos_addresses
            ]
        ],
        "model_sha256": _sha256(model_path),
        "config_sha256": _sha256(config_path),
    }
    (output_dir / "policy_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(config_path, output_dir / "training_config.yaml")


def _completed_stages_from_resume(resume_path: Path | None) -> list[str]:
    if resume_path is None:
        return []
    metadata_path = resume_path.expanduser().resolve().parent / "policy_metadata.yaml"
    if not metadata_path.exists():
        return []
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    stages = metadata.get("completed_stages", [])
    if not isinstance(stages, list) or not all(isinstance(item, str) for item in stages):
        raise ValueError(f"Invalid completed_stages in {metadata_path}")
    return list(stages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        action="append",
        help="Curriculum stage to run. Repeat to select multiple stages.",
    )
    parser.add_argument("--timesteps", type=int, help="Override each selected stage.")
    parser.add_argument("--n-envs", type=int, help="Override vector environment count.")
    parser.add_argument("--seed", type=int, help="Override experiment seed.")
    parser.add_argument("--resume", type=Path, help="PPO checkpoint to continue.")
    parser.add_argument(
        "--prior-stage",
        action="append",
        default=[],
        help="Completed stage to retain in metadata when resuming an intermediate file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "policies/latest/arm_grasp_residual",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one rollout-sized training check on the first stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ppo_config = config["ppo"]
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    n_envs = int(ppo_config["n_envs"] if args.n_envs is None else args.n_envs)
    if n_envs < 1:
        raise ValueError("n_envs must be positive")
    torch.set_num_threads(int(ppo_config.get("torch_threads", 1)))

    configured_stages = config["curriculum"]["stages"]
    stage_names = [stage["name"] for stage in configured_stages]
    selected = stage_names if not args.stage else args.stage
    unknown = sorted(set(selected) - set(stage_names))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}; available={stage_names}")
    if args.smoke:
        selected = selected[:1]

    stage_map = {stage["name"]: stage for stage in configured_stages}
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    log_dir = output_dir / "logs"
    for directory in (output_dir, checkpoint_dir, best_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model: PPO | None = None
    train_env: DummyVecEnv | None = None
    completed_stages = _completed_stages_from_resume(args.resume)
    for stage_name in args.prior_stage:
        if stage_name not in completed_stages:
            completed_stages.append(stage_name)
    total_timesteps = 0

    for stage_index, stage_name in enumerate(selected):
        if train_env is not None:
            train_env.close()
        train_env = _make_vector_env(
            config, stage_name, n_envs, seed + stage_index * 1000
        )
        eval_env = _make_vector_env(
            config, stage_name, 1, seed + 100_000 + stage_index * 1000
        )

        if model is None:
            if args.resume:
                model = PPO.load(
                    args.resume.expanduser().resolve(),
                    env=train_env,
                    device=ppo_config.get("device", "cpu"),
                )
                model.tensorboard_log = str(log_dir)
                model.learning_rate = float(ppo_config["learning_rate"])
                model.lr_schedule = FloatSchedule(model.learning_rate)
                model.ent_coef = float(ppo_config["ent_coef"])
                total_timesteps = int(model.num_timesteps)
            else:
                policy = ppo_config["policy"]
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=float(ppo_config["learning_rate"]),
                    n_steps=int(ppo_config["n_steps"]),
                    batch_size=int(ppo_config["batch_size"]),
                    n_epochs=int(ppo_config["n_epochs"]),
                    gamma=float(ppo_config["gamma"]),
                    gae_lambda=float(ppo_config["gae_lambda"]),
                    clip_range=float(ppo_config["clip_range"]),
                    ent_coef=float(ppo_config["ent_coef"]),
                    vf_coef=float(ppo_config["vf_coef"]),
                    max_grad_norm=float(ppo_config["max_grad_norm"]),
                    target_kl=float(ppo_config["target_kl"]),
                    policy_kwargs={
                        "net_arch": list(policy["net_arch"]),
                        "activation_fn": _activation(policy["activation"]),
                    },
                    tensorboard_log=str(log_dir),
                    seed=seed,
                    device=ppo_config.get("device", "cpu"),
                    verbose=1,
                )
        else:
            model.set_env(train_env)

        rollout_size = int(ppo_config["n_steps"]) * n_envs
        timesteps = int(
            stage_map[stage_name]["timesteps"]
            if args.timesteps is None
            else args.timesteps
        )
        if args.smoke:
            timesteps = rollout_size
        if timesteps < 1:
            raise ValueError("Stage timesteps must be positive")

        evaluation = config["evaluation"]
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir / stage_name),
            log_path=str(log_dir / stage_name),
            eval_freq=max(int(evaluation["frequency"]) // n_envs, 1),
            n_eval_episodes=int(evaluation["episodes"]),
            deterministic=bool(evaluation["deterministic"]),
            render=False,
        )
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(
                        int(evaluation["checkpoint_frequency"]) // n_envs, 1
                    ),
                    save_path=str(checkpoint_dir),
                    name_prefix=f"{stage_name}_ppo",
                ),
                eval_callback,
            ]
        )

        print(
            f"\n=== stage={stage_name} envs={n_envs} "
            f"requested_steps={timesteps} rollout_size={rollout_size} ==="
        )
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=stage_name,
        )
        best_model_path = best_dir / stage_name / "best_model.zip"
        if best_model_path.exists():
            model = PPO.load(
                best_model_path,
                env=train_env,
                device=ppo_config.get("device", "cpu"),
            )
            model.tensorboard_log = str(log_dir)
            model.learning_rate = float(ppo_config["learning_rate"])
            model.lr_schedule = FloatSchedule(model.learning_rate)
            model.ent_coef = float(ppo_config["ent_coef"])
            print(f"Promoted stage best model: {best_model_path}")
        total_timesteps = int(model.num_timesteps)
        if stage_name not in completed_stages:
            completed_stages.append(stage_name)
        model.save(output_dir / f"arm_grasp_{stage_name}")
        _write_metadata(output_dir, config, completed_stages, total_timesteps)

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10 if args.smoke else int(evaluation["episodes"]),
            deterministic=True,
            warn=False,
        )
        print(
            f"stage={stage_name} evaluation_reward="
            f"{mean_reward:.3f} +/- {std_reward:.3f}"
        )
        eval_env.close()

    if model is None or train_env is None:
        raise RuntimeError("No curriculum stage was selected")
    model.save(output_dir / "arm_grasp_latest")
    _write_metadata(output_dir, config, completed_stages, total_timesteps)
    train_env.close()
    print(f"Saved policy: {output_dir / 'arm_grasp_latest.zip'}")


if __name__ == "__main__":
    main()
