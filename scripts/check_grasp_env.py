"""Check environment contracts and deterministic reachability before PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from stable_baselines3.common.env_checker import check_env

from envs import load_config, make_grasp_env
from envs.factory import DEFAULT_CONFIG


def solve_pregrasp_joints(env, target: np.ndarray) -> np.ndarray:
    """Solve a level-gripper reference pose used only by this contract check."""
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


def run_reference_episode(env, seed: int) -> dict:
    _, info = env.reset(seed=seed)
    if not env.reference_control_enabled:
        target = env._pregrasp_target(info["object_initial_position"])
        final_pregrasp = solve_pregrasp_joints(env, target)

    for _ in range(env.max_episode_steps):
        if env.reference_control_enabled:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            if info["is_grasped"]:
                joint_goal = env._stay_joint_positions
            elif info["approach_stage"] < info["approach_stage_count"]:
                joint_goal = env._episode_approach_joint_waypoints[
                    info["approach_stage"]
                ]
            else:
                joint_goal = final_pregrasp
            action = np.clip(
                (joint_goal - env._arm_target) / env.action_scale,
                -1.0,
                1.0,
            )
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default="full_tower")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    config = load_config(args.config)
    env = make_grasp_env(config, stage_name=args.stage)
    check_env(env, warn=True)
    results = [run_reference_episode(env, seed) for seed in range(args.samples)]
    successes = sum(bool(info["is_success"]) for info in results)
    collisions = sum(bool(info["collision_failure"]) for info in results)
    buckets = sorted({str(info["object_sample_bucket"]) for info in results})
    env.close()

    print(f"observation/action: 33/4")
    print(f"stage: {args.stage}")
    print(f"reference success: {successes}/{args.samples}")
    print(f"collision failures: {collisions}/{args.samples}")
    print(f"covered buckets: {', '.join(buckets)}")
    if successes != args.samples or collisions:
        raise SystemExit("Grasp environment reachability check failed")


if __name__ == "__main__":
    main()
