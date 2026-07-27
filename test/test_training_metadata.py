"""Tests for PPO deployment metadata generation."""

import numpy as np
import yaml

from envs import load_config
from train.train_grasp_ppo import _write_metadata


def test_metadata_exports_named_arm_stay_positions(tmp_path):
    config = load_config("configs/arm_grasp_randomized_ppo.yaml")

    _write_metadata(tmp_path, config, [], 0)

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
    np.testing.assert_allclose(
        metadata["stay_joint_positions"],
        [0.104311, 0.027612, -0.001534, -1.638291],
    )
