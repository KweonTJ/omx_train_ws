"""Contract tests for randomized grasp training."""

import numpy as np
from stable_baselines3.common.env_checker import check_env

from envs import load_config, make_grasp_env


def test_environment_contract_and_random_rollout():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    env = make_grasp_env(config, stage_name="full_tower")
    check_env(env, warn=True)
    observation, _ = env.reset(seed=7)

    assert observation.shape == (33,)
    assert env.action_space.shape == (4,)
    assert env.observation_space.contains(observation)

    for _ in range(100):
        observation, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()


def test_randomized_object_pose_stays_on_tower_and_covers_buckets():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    env = make_grasp_env(config, stage_name="full_tower")
    buckets = set()

    for seed in range(300):
        _, info = env.reset(seed=seed)
        position = info["object_initial_position"]
        yaw = info["object_yaw"]
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        footprint = np.array(
            [
                c * env._object_half_size[0] + s * env._object_half_size[1],
                s * env._object_half_size[0] + c * env._object_half_size[1],
            ]
        )
        relative = np.abs(position[:2] - env._tower_center[:2])
        assert np.all(
            relative + footprint + env.support_clearance
            <= env._tower_half_size[:2] + 1e-9
        )
        buckets.add(info["object_sample_bucket"])

    assert len(buckets) >= 7
    env.close()


def test_fixed_pose_option_is_reproducible():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    env = make_grasp_env(config, stage_name="full_tower")
    requested = np.array([0.27, 0.01, 0.1975])
    _, info = env.reset(
        seed=123,
        options={"object_position": requested, "object_yaw": 0.2},
    )

    np.testing.assert_allclose(info["object_initial_position"], requested)
    assert np.isclose(info["object_yaw"], 0.2)
    env.close()


def test_zero_residual_reference_controller_completes_irregular_grasps():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    env = make_grasp_env(config, stage_name="full_tower")
    zero_residual = np.zeros(4, dtype=np.float32)

    for seed in (0, 3, 7, 12, 19):
        _, info = env.reset(seed=seed)
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, info = env.step(zero_residual)
            if terminated or truncated:
                break
        assert info["is_success"], (seed, info)
        assert not info["collision_failure"], (seed, info)

    env.close()


def test_zero_residual_reference_controller_places_and_returns_to_stay():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    env = make_grasp_env(config, stage_name="place_full_tower")
    zero_residual = np.zeros(4, dtype=np.float32)

    for seed in (0, 7, 11, 19):
        _, info = env.reset(seed=seed)
        assert info["episode_task"] == "place"
        assert info["is_grasped"]
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, info = env.step(zero_residual)
            if terminated or truncated:
                break
        assert info["is_success"], (seed, info)
        assert info["released"], (seed, info)
        assert info["placement_error"] <= env.placement_tolerance
        assert not info["collision_failure"], (seed, info)

    env.close()


def test_mixed_and_domain_randomized_stages_cover_contracts():
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    mixed = make_grasp_env(config, stage_name="pick_place_mixed")
    tasks = {mixed.reset(seed=seed)[1]["episode_task"] for seed in range(40)}
    assert tasks == {"pick", "place"}
    mixed.close()

    randomized = make_grasp_env(config, stage_name="sim2real_robust")
    _, info = randomized.reset(seed=4)
    domain = info["domain_randomization"]
    assert len(domain["tower_offset"]) == 2
    assert 0 <= domain["action_delay_steps"] <= 2
    assert 0.90 <= domain["object_scale"] <= 1.10
    assert 0.80 <= domain["damping_scale"] <= 1.20
    assert 0.85 <= domain["gain_scale"] <= 1.15
    assert 0.70 <= domain["friction_scale"] <= 1.30
    randomized.close()
