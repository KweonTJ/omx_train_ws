"""Tests for PPO deployment metadata generation."""

import numpy as np
import yaml
from stable_baselines3 import PPO

from envs import load_config
from train.train_grasp_ppo import _preserve_sample_counter, _write_metadata


def test_metadata_exports_named_arm_stay_positions(tmp_path):
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")

    _write_metadata(tmp_path, config, ["delivery_full_tower"], 0)

    metadata = yaml.safe_load(
        (tmp_path / "policy_metadata.yaml").read_text(encoding="utf-8")
    )
    assert metadata["joint_names"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
    ]
    assert metadata["action_schema_version"] == 3
    assert metadata["control_mode"] == (
        "reference_plus_residual_with_joint4_pitch_coupling"
    )
    assert metadata["gripper_pitch_hold_enabled"]
    assert metadata["gripper_pitch_target"] == 0.0
    assert metadata["gripper_pitch_joint4_max_step"] == 0.028
    assert metadata["supported_commands"] == ["PICK", "PLACE"]
    assert metadata["delivery_sequence_trained"]
    assert metadata["transport_hold_steps"] == 50
    assert metadata["delivery_stage_transport_hold_steps"] == {
        "delivery_center": 25,
        "delivery_full_tower": 50,
        "delivery_sim2real_robust": 75,
    }
    np.testing.assert_allclose(
        metadata["stay_joint_positions"],
        [0.104311, 0.027612, -0.001534, -1.638291],
    )


def test_promoted_stage_counter_can_preserve_all_sampled_transitions(tmp_path):
    """Regression contract for restoring a best checkpoint between stages."""
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")
    from envs import make_grasp_env

    env = make_grasp_env(config, stage_name="delivery_center")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
    )
    model.num_timesteps = 1234
    checkpoint = tmp_path / "best_model"
    model.save(checkpoint)
    promoted = PPO.load(checkpoint, env=env, device="cpu")
    assert promoted.num_timesteps == 1234

    stage_end_timesteps = 5678
    promoted = _preserve_sample_counter(promoted, stage_end_timesteps)
    assert promoted.num_timesteps == 5678
    env.close()
