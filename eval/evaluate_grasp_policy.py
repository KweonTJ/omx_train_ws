"""Evaluate grasp success by randomized object-position bucket."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
import yaml

from envs import load_config, make_grasp_env
from envs.factory import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "policies/latest/arm_grasp/arm_grasp_latest.zip",
    )
    parser.add_argument("--stage", default="full_tower")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    config = load_config(args.config)
    env = make_grasp_env(config, stage_name=args.stage)
    model = PPO.load(args.policy.expanduser().resolve(), device="cpu")
    rng = np.random.default_rng(args.seed)

    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"episodes": 0.0, "successes": 0.0, "collisions": 0.0}
    )
    tasks: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "episodes": 0.0,
            "successes": 0.0,
            "collisions": 0.0,
            "grasped_final": 0.0,
            "released_final": 0.0,
        }
    )
    total_reward = 0.0
    episode_lengths: list[int] = []
    grasped_count = 0
    released_count = 0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        bucket = str(info["object_sample_bucket"])
        episode_reward = 0.0
        final_info: dict[str, Any] = info
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_reward += float(reward)
            if terminated or truncated:
                break

        success = bool(final_info["is_success"])
        collision = bool(final_info.get("collision_failure", False))
        task = str(final_info["episode_task"])
        buckets[bucket]["episodes"] += 1
        buckets[bucket]["successes"] += int(success)
        buckets[bucket]["collisions"] += int(collision)
        tasks[task]["episodes"] += 1
        tasks[task]["successes"] += int(success)
        tasks[task]["collisions"] += int(collision)
        grasped_final = bool(final_info["is_grasped"])
        released_final = bool(final_info.get("released", False))
        tasks[task]["grasped_final"] += int(grasped_final)
        tasks[task]["released_final"] += int(released_final)
        grasped_count += int(grasped_final)
        released_count += int(released_final)
        total_reward += episode_reward
        episode_lengths.append(step + 1)

    bucket_report = {}
    for bucket, values in sorted(buckets.items()):
        episodes = int(values["episodes"])
        bucket_report[bucket] = {
            "episodes": episodes,
            "success_rate": values["successes"] / episodes,
            "collision_rate": values["collisions"] / episodes,
        }

    task_report = {}
    for task, values in sorted(tasks.items()):
        episodes = int(values["episodes"])
        task_report[task] = {
            "episodes": episodes,
            "success_rate": values["successes"] / episodes,
            "collision_rate": values["collisions"] / episodes,
            "grasped_final_rate": values["grasped_final"] / episodes,
            "released_final_rate": values["released_final"] / episodes,
        }

    report = {
        "policy": str(args.policy.expanduser().resolve()),
        "stage": args.stage,
        "episodes": args.episodes,
        "success_rate": sum(item["successes"] for item in buckets.values())
        / args.episodes,
        "grasped_final_rate": grasped_count / args.episodes,
        "released_final_rate": released_count / args.episodes,
        "collision_rate": sum(item["collisions"] for item in buckets.values())
        / args.episodes,
        "mean_reward": total_reward / args.episodes,
        "mean_episode_steps": float(np.mean(episode_lengths)),
        "task_breakdown": task_report,
        "position_buckets": bucket_report,
    }
    text = yaml.safe_dump(report, sort_keys=False)
    print(text)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
