"""Replay a trained grasp policy in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from envs import load_config, make_grasp_env
from envs.factory import DEFAULT_CONFIG


DEFAULT_POLICY = (
    ROOT / "policies/latest/arm_delivery_residual_v2/arm_grasp_latest.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--stage", default="sim2real_robust")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--pause", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pause < 0.0:
        raise ValueError("pause must be non-negative")

    config = load_config(args.config)
    env = make_grasp_env(config, stage_name=args.stage)
    policy_path = args.policy.expanduser().resolve()
    model = PPO.load(policy_path, device="cpu")
    rng = np.random.default_rng(args.seed)
    control_period = env.model.opt.timestep * env.frame_skip

    print(f"Policy: {policy_path}")
    print(f"Stage: {args.stage}")
    print("Close the MuJoCo window to stop replay.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.10, 0.0, 0.16])
        viewer.cam.distance = 1.05
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        episode = 0
        while viewer.is_running():
            episode += 1
            episode_seed = int(rng.integers(0, 2**31 - 1))
            observation, info = env.reset(seed=episode_seed)
            viewer.sync()
            print(
                f"episode={episode} seed={episode_seed} "
                f"task={info['episode_task']} "
                f"bucket={info['object_sample_bucket']} "
                f"position={np.round(info['object_initial_position'], 4)}"
            )

            final_info = info
            for _ in range(env.max_episode_steps):
                if not viewer.is_running():
                    break
                started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, final_info = env.step(action)
                viewer.sync()
                remaining = control_period - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
                if terminated or truncated:
                    break

            print(
                f"episode={episode} success={final_info['is_success']} "
                f"grasped={final_info['is_grasped']} "
                f"released={final_info.get('released', False)} "
                f"collision={final_info.get('collision_failure', False)} "
                f"steps={final_info['elapsed_steps']}"
            )
            pause_until = time.perf_counter() + args.pause
            while viewer.is_running() and time.perf_counter() < pause_until:
                viewer.sync()
                time.sleep(min(0.02, pause_until - time.perf_counter()))

    env.close()


if __name__ == "__main__":
    main()
