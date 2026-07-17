"""Randomized object-pose grasp and return-to-stay task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.optimize import least_squares


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ArmRandomizedGraspEnv(gym.Env):
    """Train joint1-joint4 to grasp a randomized box and return to stay."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    PICK_REACH = 0
    PICK_TO_STAY = 1
    PLACE_REACH = 2
    PLACE_TO_STAY = 3
    OBSERVATION_SIZE = 33

    ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4")
    BODY_COLLISION_GEOMS = (
        "base",
        "camera_plate",
        "camera_body",
        "camera_mount",
        "support_front_left",
        "support_front_right",
        "support_back_left",
        "support_back_right",
        "upper_waffle_plate",
        "link1_collision",
        "task_tower",
    )
    ARM_COLLISION_GEOMS = (
        "link2_mesh",
        "link3_mesh",
        "link4_mesh",
        "link5_mesh",
        "gripper_left_mesh",
        "gripper_right_mesh",
    )

    def __init__(
        self,
        model_path: str | Path,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 400,
        action_scale: tuple[float, ...] | list[float] = (0.014, 0.014, 0.014, 0.014),
        action_filter_coefficient: float = 0.18,
        reference_control_enabled: bool = True,
        residual_action_scale: float = 0.10,
        reference_action_limit: float = 1.0,
        final_approach_action_limit: float = 1.0,
        object_center: tuple[float, ...] | list[float] = (0.27, 0.0, 0.1975),
        object_position_range: tuple[float, ...] | list[float] = (0.024, 0.024, 0.0),
        object_yaw_range: float = np.pi,
        edge_sample_probability: float = 0.4,
        edge_min_fraction: float = 0.7,
        support_clearance: float = 0.001,
        initial_joint_noise: tuple[float, ...] | list[float] = (0.01, 0.01, 0.01, 0.01),
        vision_position_noise_std: tuple[float, ...] | list[float] = (0.0, 0.0, 0.0),
        vision_yaw_noise_std: float = 0.0,
        vision_update_interval_steps: int = 1,
        vision_dropout_probability: float = 0.0,
        vision_filter_coefficient: float = 0.35,
        start_grasped_probability: float = 0.0,
        place_episode_probability: float = 0.0,
        start_released_probability: float = 0.0,
        grasped_start_joint_positions: tuple[float, ...] | list[float] = (
            0.0,
            1.15968,
            -0.48813,
            -0.67155,
        ),
        approach_joint_waypoints: tuple[tuple[float, ...], ...] | list[list[float]] = (
            (0.0, -0.5, 0.5, 0.0),
            (0.0, 0.5, 0.2, -0.7),
        ),
        approach_waypoint_tolerance: float = 0.08,
        pregrasp_height_offset: float = 0.025,
        gripper_open: float = 0.019,
        gripper_close: float = -0.010,
        gripper_open_tolerance: float = 0.003,
        close_distance: float = 0.042,
        close_xy_distance: float = 0.035,
        close_z_tolerance: float = 0.030,
        close_bearing_tolerance: float = 0.35,
        close_roll_tolerance: float = 0.35,
        close_stable_steps: int = 4,
        close_settle_steps: int = 12,
        max_closing_steps: int = 60,
        attach_distance: float = 0.048,
        attach_xy_distance: float = 0.040,
        attach_z_tolerance: float = 0.035,
        release_stable_steps: int = 4,
        release_settle_steps: int = 12,
        placement_tolerance: float = 0.015,
        stay_tolerance: float = 0.050,
        minimum_lift_height: float = 0.025,
        minimum_object_clearance: float = 0.015,
        success_stable_steps: int = 8,
        collision_termination_depth: float = 0.004,
        joint_velocity_scale: tuple[float, ...] | list[float] = (2.0, 2.0, 2.0, 2.0),
        workspace_min: tuple[float, ...] | list[float] = (0.10, -0.18, 0.08),
        workspace_max: tuple[float, ...] | list[float] = (0.42, 0.18, 0.52),
        tower_position_range: tuple[float, ...] | list[float] = (0.0, 0.0),
        tower_height_range: float = 0.0,
        object_size_scale_range: tuple[float, ...] | list[float] = (1.0, 1.0),
        arm_damping_scale_range: tuple[float, ...] | list[float] = (1.0, 1.0),
        actuator_gain_scale_range: tuple[float, ...] | list[float] = (1.0, 1.0),
        friction_scale_range: tuple[float, ...] | list[float] = (1.0, 1.0),
        maximum_action_delay_steps: int = 0,
        reward: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.model_path = Path(model_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)

        self.action_scale = self._vector(action_scale, "action_scale", 4)
        self.action_filter_coefficient = float(action_filter_coefficient)
        self.reference_control_enabled = bool(reference_control_enabled)
        self.residual_action_scale = float(residual_action_scale)
        self.reference_action_limit = float(reference_action_limit)
        self.final_approach_action_limit = float(final_approach_action_limit)
        self.object_center = self._vector(object_center, "object_center", 3)
        self.object_position_range = self._vector(
            object_position_range, "object_position_range", 3
        )
        self.object_yaw_range = float(object_yaw_range)
        self.edge_sample_probability = float(edge_sample_probability)
        self.edge_min_fraction = float(edge_min_fraction)
        self.support_clearance = float(support_clearance)
        self.initial_joint_noise = self._vector(
            initial_joint_noise, "initial_joint_noise", 4
        )
        self.vision_position_noise_std = self._vector(
            vision_position_noise_std, "vision_position_noise_std", 3
        )
        self.vision_yaw_noise_std = float(vision_yaw_noise_std)
        self.vision_update_interval_steps = int(vision_update_interval_steps)
        self.vision_dropout_probability = float(vision_dropout_probability)
        self.vision_filter_coefficient = float(vision_filter_coefficient)
        self.start_grasped_probability = float(start_grasped_probability)
        self.place_episode_probability = float(place_episode_probability)
        self.start_released_probability = float(start_released_probability)
        self.grasped_start_joint_positions = self._vector(
            grasped_start_joint_positions, "grasped_start_joint_positions", 4
        )
        self.approach_joint_waypoints = np.asarray(
            approach_joint_waypoints, dtype=np.float64
        )
        self.approach_waypoint_tolerance = float(approach_waypoint_tolerance)
        self.pregrasp_height_offset = float(pregrasp_height_offset)

        self.gripper_open = float(gripper_open)
        self.gripper_close = float(gripper_close)
        self.gripper_open_tolerance = float(gripper_open_tolerance)
        self.close_distance = float(close_distance)
        self.close_xy_distance = float(close_xy_distance)
        self.close_z_tolerance = float(close_z_tolerance)
        self.close_bearing_tolerance = float(close_bearing_tolerance)
        self.close_roll_tolerance = float(close_roll_tolerance)
        self.close_stable_steps = int(close_stable_steps)
        self.close_settle_steps = int(close_settle_steps)
        self.max_closing_steps = int(max_closing_steps)
        self.attach_distance = float(attach_distance)
        self.attach_xy_distance = float(attach_xy_distance)
        self.attach_z_tolerance = float(attach_z_tolerance)
        self.release_stable_steps = int(release_stable_steps)
        self.release_settle_steps = int(release_settle_steps)
        self.placement_tolerance = float(placement_tolerance)
        self.stay_tolerance = float(stay_tolerance)
        self.minimum_lift_height = float(minimum_lift_height)
        self.minimum_object_clearance = float(minimum_object_clearance)
        self.success_stable_steps = int(success_stable_steps)
        self.collision_termination_depth = float(collision_termination_depth)
        self.joint_velocity_scale = self._vector(
            joint_velocity_scale, "joint_velocity_scale", 4
        )
        self.workspace_min = self._vector(workspace_min, "workspace_min", 3)
        self.workspace_max = self._vector(workspace_max, "workspace_max", 3)
        self.tower_position_range = self._vector(
            tower_position_range, "tower_position_range", 2
        )
        self.tower_height_range = float(tower_height_range)
        self.object_size_scale_range = self._range(
            object_size_scale_range, "object_size_scale_range"
        )
        self.arm_damping_scale_range = self._range(
            arm_damping_scale_range, "arm_damping_scale_range"
        )
        self.actuator_gain_scale_range = self._range(
            actuator_gain_scale_range, "actuator_gain_scale_range"
        )
        self.friction_scale_range = self._range(
            friction_scale_range, "friction_scale_range"
        )
        self.maximum_action_delay_steps = int(maximum_action_delay_steps)

        default_reward = {
            "reach_progress": 120.0,
            "reach_distance": 2.0,
            "bearing_error": 0.20,
            "roll_error": 0.50,
            "approach_joint_progress": 12.0,
            "approach_joint_error": 0.50,
            "approach_waypoint_bonus": 5.0,
            "close_ready_bonus": 2.0,
            "grasp_bonus": 15.0,
            "release_bonus": 15.0,
            "placed": 0.20,
            "stay_progress": 30.0,
            "stay_error": 1.5,
            "holding": 0.20,
            "lift_progress": 20.0,
            "action_delta": 3.0,
            "action_jerk": 4.0,
            "residual_action": 0.02,
            "joint_velocity": 0.003,
            "joint_limit": 0.25,
            "collision": 8.0,
            "collision_failure": 1500.0,
            "failed_close": 3.0,
            "time": 0.01,
            "success": 200.0,
        }
        self.reward_weights = default_reward
        if reward:
            self.reward_weights.update(
                {str(key): float(value) for key, value in reward.items()}
            )

        self._validate_parameters()
        self._bind_model_names()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self._renderer: mujoco.Renderer | None = None
        self._elapsed_steps = 0
        self._phase = self.PICK_REACH
        self._arm_target = np.zeros(4, dtype=np.float64)
        self._filtered_action = np.zeros(4, dtype=np.float64)
        self._previous_action = np.zeros(4, dtype=np.float64)
        self._previous_filtered_delta = np.zeros(4, dtype=np.float64)
        self._gripper_target = self.gripper_open
        self._object_initial_position = self.object_center.copy()
        self._place_target_position = self.object_center.copy()
        self._episode_object_center = self.object_center.copy()
        self._episode_task = "pick"
        self._reference_pregrasp_joint_positions = np.zeros(4, dtype=np.float64)
        self._object_yaw = 0.0
        self._object_sample_bucket = "center"
        self._vision_position = self.object_center.copy()
        self._vision_yaw = 0.0
        self._vision_valid = True
        self._vision_dropout_count = 0
        self._started_grasped = False
        self._started_released = False
        self._approach_stage = 0
        self._approach_just_advanced = False
        self._episode_approach_joint_waypoints = np.empty((0, 4))
        self._episode_approach_eef_targets = np.empty((0, 3))
        self._close_gate_count = 0
        self._close_just_triggered = False
        self._closing = False
        self._closing_steps = 0
        self._grasped = False
        self._grasp_just_latched = False
        self._failed_close = False
        self._release_gate_count = 0
        self._release_just_triggered = False
        self._release_just_completed = False
        self._releasing = False
        self._release_steps = 0
        self._grasp_attach_offset = np.zeros(3, dtype=np.float64)
        self._action_delay_steps = 0
        self._action_delay_queue: list[np.ndarray] = []
        self._episode_randomization: dict[str, float | list[float]] = {}
        self._success_count = 0
        self._previous_reach_distance = 0.0
        self._previous_approach_joint_error = 0.0
        self._previous_stay_error = 0.0
        self._previous_object_height = 0.0

    @staticmethod
    def _vector(value: Any, name: str, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {size} finite values")
        return array

    @classmethod
    def _range(cls, value: Any, name: str) -> np.ndarray:
        interval = cls._vector(value, name, 2)
        if interval[0] <= 0.0 or interval[1] < interval[0]:
            raise ValueError(f"{name} must be a positive [minimum, maximum]")
        return interval

    def _validate_parameters(self) -> None:
        if self.model.nu != 5:
            raise ValueError(f"Expected five actuators, got {self.model.nu}")
        if self.frame_skip < 1 or self.max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        if not 0.0 < self.action_filter_coefficient <= 1.0:
            raise ValueError("action_filter_coefficient must be in (0, 1]")
        if not 0.0 <= self.residual_action_scale <= 1.0:
            raise ValueError("residual_action_scale must be in [0, 1]")
        if not 0.0 < self.reference_action_limit <= 1.0:
            raise ValueError("reference_action_limit must be in (0, 1]")
        if not 0.0 < self.final_approach_action_limit <= 1.0:
            raise ValueError("final_approach_action_limit must be in (0, 1]")
        if np.any(self.action_scale <= 0.0):
            raise ValueError("action_scale must be positive")
        if np.any(self.object_position_range < 0.0):
            raise ValueError("object_position_range cannot be negative")
        if not 0.0 <= self.edge_sample_probability <= 1.0:
            raise ValueError("edge_sample_probability must be in [0, 1]")
        if not 0.0 <= self.edge_min_fraction <= 1.0:
            raise ValueError("edge_min_fraction must be in [0, 1]")
        if not 0.0 <= self.vision_dropout_probability <= 1.0:
            raise ValueError("vision_dropout_probability must be in [0, 1]")
        if not 0.0 < self.vision_filter_coefficient <= 1.0:
            raise ValueError("vision_filter_coefficient must be in (0, 1]")
        if not 0.0 <= self.start_grasped_probability <= 1.0:
            raise ValueError("start_grasped_probability must be in [0, 1]")
        if not 0.0 <= self.place_episode_probability <= 1.0:
            raise ValueError("place_episode_probability must be in [0, 1]")
        if not 0.0 <= self.start_released_probability <= 1.0:
            raise ValueError("start_released_probability must be in [0, 1]")
        if self.vision_update_interval_steps < 1:
            raise ValueError("vision_update_interval_steps must be positive")
        if (
            self.approach_joint_waypoints.ndim != 2
            or self.approach_joint_waypoints.shape[1] != 4
            or not np.isfinite(self.approach_joint_waypoints).all()
        ):
            raise ValueError("approach_joint_waypoints must have shape (N, 4)")
        if self.approach_waypoint_tolerance <= 0.0:
            raise ValueError("approach_waypoint_tolerance must be positive")
        if np.any(self.workspace_max <= self.workspace_min):
            raise ValueError("workspace_max must be greater than workspace_min")
        if (
            self.close_stable_steps < 1
            or self.release_stable_steps < 1
            or self.success_stable_steps < 1
        ):
            raise ValueError("stable step counts must be positive")
        if self.release_settle_steps < 1 or self.placement_tolerance <= 0.0:
            raise ValueError("release settling and placement tolerance must be positive")
        if self.minimum_object_clearance < 0.0:
            raise ValueError("minimum_object_clearance cannot be negative")
        if np.any(self.tower_position_range < 0.0) or self.tower_height_range < 0.0:
            raise ValueError("tower randomization ranges cannot be negative")
        if self.maximum_action_delay_steps < 0:
            raise ValueError("maximum_action_delay_steps cannot be negative")

    def _bind_model_names(self) -> None:
        self._arm_joint_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.ARM_JOINTS
        )
        self._arm_qpos_addr = np.array(
            [self.model.jnt_qposadr[index] for index in self._arm_joint_ids],
            dtype=np.int32,
        )
        self._arm_qvel_addr = np.array(
            [self.model.jnt_dofadr[index] for index in self._arm_joint_ids],
            dtype=np.int32,
        )
        gripper_joint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_joint"
        )
        self._gripper_qpos_addr = int(self.model.jnt_qposadr[gripper_joint_id])
        self._gripper_qvel_addr = int(self.model.jnt_dofadr[gripper_joint_id])
        right_gripper_joint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT, "gripper_right_joint"
        )
        self._right_gripper_qpos_addr = int(
            self.model.jnt_qposadr[right_gripper_joint_id]
        )
        self._eef_site_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        self._object_site_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SITE, "grasp_object_site"
        )
        self._object_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM, "grasp_object"
        )
        self._tower_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM, "task_tower"
        )
        self._arm_base_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "link2"
        )
        self._stay_key_id = self._name_id(mujoco.mjtObj.mjOBJ_KEY, "stay")

        object_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "grasp_object_body"
        )
        target_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "reach_target_body"
        )
        self._object_mocap_id = int(self.model.body_mocapid[object_body_id])
        self._target_mocap_id = int(self.model.body_mocapid[target_body_id])
        if self._object_mocap_id < 0 or self._target_mocap_id < 0:
            raise ValueError("Object and reach target bodies must be mocap bodies")

        self._ctrl_low = self.model.actuator_ctrlrange[:4, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:4, 1].copy()
        self._stay_joint_positions = self.model.key_qpos[
            self._stay_key_id, self._arm_qpos_addr
        ].copy()
        self._tower_center = self.model.geom_pos[self._tower_geom_id].copy()
        self._tower_half_size = self.model.geom_size[self._tower_geom_id].copy()
        self._object_half_size = self.model.geom_size[self._object_geom_id].copy()
        self._tower_nominal_center = self._tower_center.copy()
        self._tower_nominal_half_size = self._tower_half_size.copy()
        self._object_nominal_half_size = self._object_half_size.copy()
        self._tower_nominal_friction = self.model.geom_friction[
            self._tower_geom_id
        ].copy()
        self._object_nominal_friction = self.model.geom_friction[
            self._object_geom_id
        ].copy()
        self._arm_nominal_damping = self.model.dof_damping[
            self._arm_qvel_addr
        ].copy()
        self._arm_nominal_gain = self.model.actuator_gainprm[:4, 0].copy()
        self._arm_nominal_bias = self.model.actuator_biasprm[:4, 1].copy()

        self._body_collision_geom_ids = set(
            self._named_ids(mujoco.mjtObj.mjOBJ_GEOM, self.BODY_COLLISION_GEOMS)
        )
        self._arm_collision_geom_ids = set(
            self._named_ids(mujoco.mjtObj.mjOBJ_GEOM, self.ARM_COLLISION_GEOMS)
        )

        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, self._stay_key_id)
        mujoco.mj_forward(self.model, scratch)
        self._stay_eef_position = scratch.site_xpos[self._eef_site_id].copy()

    def _apply_domain_randomization(self) -> None:
        tower_offset = self.np_random.uniform(
            -self.tower_position_range, self.tower_position_range
        )
        height_offset = float(
            self.np_random.uniform(-self.tower_height_range, self.tower_height_range)
        )
        tower_height = 2.0 * self._tower_nominal_half_size[2] + height_offset
        if tower_height <= 2.0 * self._object_nominal_half_size[2]:
            raise ValueError("Randomized tower height is too small")

        object_scale = float(self.np_random.uniform(*self.object_size_scale_range))
        damping_scale = float(self.np_random.uniform(*self.arm_damping_scale_range))
        gain_scale = float(self.np_random.uniform(*self.actuator_gain_scale_range))
        friction_scale = float(self.np_random.uniform(*self.friction_scale_range))

        self.model.geom_pos[self._tower_geom_id] = self._tower_nominal_center
        self.model.geom_pos[self._tower_geom_id, :2] += tower_offset
        self.model.geom_pos[self._tower_geom_id, 2] = 0.5 * tower_height
        self.model.geom_size[self._tower_geom_id] = self._tower_nominal_half_size
        self.model.geom_size[self._tower_geom_id, 2] = 0.5 * tower_height
        self.model.geom_size[self._object_geom_id] = (
            self._object_nominal_half_size * object_scale
        )
        self.model.geom_friction[self._tower_geom_id] = (
            self._tower_nominal_friction * friction_scale
        )
        self.model.geom_friction[self._object_geom_id] = (
            self._object_nominal_friction * friction_scale
        )
        self.model.dof_damping[self._arm_qvel_addr] = (
            self._arm_nominal_damping * damping_scale
        )
        self.model.actuator_gainprm[:4, 0] = self._arm_nominal_gain * gain_scale
        self.model.actuator_biasprm[:4, 1] = self._arm_nominal_bias * gain_scale
        mujoco.mj_setConst(self.model, self.data)

        self._tower_center = self.model.geom_pos[self._tower_geom_id].copy()
        self._tower_half_size = self.model.geom_size[self._tower_geom_id].copy()
        self._object_half_size = self.model.geom_size[self._object_geom_id].copy()
        relative_xy = self.object_center[:2] - self._tower_nominal_center[:2]
        self._episode_object_center = np.array(
            [
                self._tower_center[0] + relative_xy[0],
                self._tower_center[1] + relative_xy[1],
                tower_height + self._object_half_size[2],
            ],
            dtype=np.float64,
        )
        self._action_delay_steps = int(
            self.np_random.integers(0, self.maximum_action_delay_steps + 1)
        )
        self._action_delay_queue = [
            np.zeros(4, dtype=np.float64) for _ in range(self._action_delay_steps)
        ]
        self._episode_randomization = {
            "tower_offset": tower_offset.tolist(),
            "tower_height": tower_height,
            "object_scale": object_scale,
            "damping_scale": damping_scale,
            "gain_scale": gain_scale,
            "friction_scale": friction_scale,
            "action_delay_steps": self._action_delay_steps,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._apply_domain_randomization()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stay_key_id)

        joint_noise = self.np_random.uniform(
            -self.initial_joint_noise, self.initial_joint_noise
        )
        arm_qpos = np.clip(
            self._stay_joint_positions + joint_noise,
            self._ctrl_low,
            self._ctrl_high,
        )
        self.data.qpos[self._arm_qpos_addr] = arm_qpos
        self.data.qvel[:] = 0.0
        self._arm_target = arm_qpos.copy()
        self._gripper_target = self.gripper_open
        self.data.ctrl[:4] = self._arm_target
        self.data.ctrl[4] = self._gripper_target

        object_position, object_yaw, bucket = self._sample_object_pose(options)
        self._place_target_position = object_position.copy()
        self._object_initial_position = object_position.copy()
        self._object_yaw = object_yaw
        self._object_sample_bucket = bucket
        self._set_object_pose(object_position, object_yaw)

        requested_task = None if options is None else options.get("task")
        if requested_task not in (None, "pick", "place"):
            raise ValueError("options['task'] must be 'pick' or 'place'")
        self._episode_task = str(
            requested_task
            or (
                "place"
                if self.np_random.random() < self.place_episode_probability
                else "pick"
            )
        )

        self._elapsed_steps = 0
        self._phase = self.PICK_REACH
        self._filtered_action.fill(0.0)
        self._previous_action.fill(0.0)
        self._previous_filtered_delta.fill(0.0)
        self._vision_position = object_position.copy()
        self._vision_yaw = object_yaw
        self._vision_valid = True
        self._vision_dropout_count = 0
        self._approach_stage = 0
        self._approach_just_advanced = False
        self._close_gate_count = 0
        self._close_just_triggered = False
        self._closing = False
        self._closing_steps = 0
        self._grasped = False
        self._grasp_just_latched = False
        self._failed_close = False
        self._release_gate_count = 0
        self._release_just_triggered = False
        self._release_just_completed = False
        self._releasing = False
        self._release_steps = 0
        self._grasp_attach_offset.fill(0.0)
        self._success_count = 0
        self._started_grasped = False
        self._started_released = False

        mujoco.mj_forward(self.model, self.data)
        if self._episode_task == "place":
            self._initialize_place_start(object_position)
        else:
            self._started_grasped = bool(
                self.np_random.random() < self.start_grasped_probability
            )
            if self._started_grasped:
                self._initialize_grasped_start(object_position)

        self._update_vision(force=True)
        self._set_reach_target(self._pregrasp_target(self._vision_position))
        self._configure_approach_waypoints(self._vision_position)
        self._reference_pregrasp_joint_positions = (
            self._solve_reference_pregrasp(self._vision_position)
        )
        if self._started_released:
            self._approach_stage = len(self._episode_approach_joint_waypoints)

        initial_info = self._get_info()
        self._previous_reach_distance = float(initial_info["reach_distance"])
        self._previous_approach_joint_error = float(
            initial_info["approach_joint_error"]
        )
        self._previous_stay_error = float(initial_info["stay_error"])
        self._previous_object_height = float(initial_info["object_position"][2])
        return self._get_observation(), initial_info

    def _initialize_place_start(self, target_position: np.ndarray) -> None:
        grasp_joints = self._solve_reference_pregrasp(target_position)
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, self._stay_key_id)
        scratch.qpos[self._arm_qpos_addr] = grasp_joints
        scratch.ctrl[:4] = grasp_joints
        mujoco.mj_forward(self.model, scratch)
        grasp_rotation = scratch.site_xmat[self._eef_site_id].reshape((3, 3))
        self._grasp_attach_offset = grasp_rotation.T @ (
            target_position - scratch.site_xpos[self._eef_site_id]
        )

        self._started_released = bool(
            self.np_random.random() < self.start_released_probability
        )
        if self._started_released:
            arm_qpos = np.clip(
                grasp_joints
                + self.np_random.uniform(-self.initial_joint_noise, self.initial_joint_noise),
                self._ctrl_low,
                self._ctrl_high,
            )
            self.data.qpos[self._arm_qpos_addr] = arm_qpos
            self.data.qpos[self._gripper_qpos_addr] = self.gripper_open
            self.data.qpos[self._right_gripper_qpos_addr] = self.gripper_open
            self._arm_target = arm_qpos.copy()
            self._gripper_target = self.gripper_open
            self._phase = self.PLACE_TO_STAY
            self._grasped = False
            self._set_object_pose(target_position, self._object_yaw)
        else:
            self.data.qpos[self._gripper_qpos_addr] = self.gripper_close
            self.data.qpos[self._right_gripper_qpos_addr] = self.gripper_close
            self._gripper_target = self.gripper_close
            self._phase = self.PLACE_REACH
            self._grasped = True

        self.data.qvel[:] = 0.0
        self.data.ctrl[:4] = self._arm_target
        self.data.ctrl[4] = self._gripper_target
        mujoco.mj_forward(self.model, self.data)
        if self._grasped:
            self._set_object_pose(self._attached_object_position(), self._object_yaw)
            mujoco.mj_forward(self.model, self.data)
        self._object_initial_position = self.data.site_xpos[
            self._object_site_id
        ].copy()

    def _initialize_grasped_start(self, object_position: np.ndarray) -> None:
        arm_qpos = self.grasped_start_joint_positions.copy()
        arm_qpos[0] = self._bearing(object_position)
        arm_qpos += self.np_random.uniform(
            -self.initial_joint_noise, self.initial_joint_noise
        )
        arm_qpos = np.clip(arm_qpos, self._ctrl_low, self._ctrl_high)
        self.data.qpos[self._arm_qpos_addr] = arm_qpos
        self.data.qpos[self._gripper_qpos_addr] = self.gripper_close
        self.data.qpos[self._right_gripper_qpos_addr] = self.gripper_close
        self.data.qvel[:] = 0.0
        self._arm_target = arm_qpos.copy()
        self._gripper_target = self.gripper_close
        self.data.ctrl[:4] = self._arm_target
        self.data.ctrl[4] = self._gripper_target
        self._phase = self.PICK_TO_STAY
        self._approach_stage = len(self._episode_approach_joint_waypoints)
        self._grasped = True
        mujoco.mj_forward(self.model, self.data)

        eef_rotation = self.data.site_xmat[self._eef_site_id].reshape((3, 3))
        world_offset = (
            self.data.site_xpos[self._object_site_id]
            - self.data.site_xpos[self._eef_site_id]
        )
        self._grasp_attach_offset = eef_rotation.T @ world_offset

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (4,) or not np.isfinite(raw_action).all():
            raise ValueError("action must contain four finite values")
        raw_action = np.clip(raw_action, -1.0, 1.0)
        was_grasped = self._grasped
        self._grasp_just_latched = False
        self._failed_close = False
        self._close_just_triggered = False
        self._release_just_triggered = False
        self._release_just_completed = False
        self._approach_just_advanced = False

        policy_action = raw_action
        if self._action_delay_steps:
            self._action_delay_queue.append(raw_action.copy())
            policy_action = self._action_delay_queue.pop(0)

        control_action = policy_action
        if self.reference_control_enabled:
            control_action = np.clip(
                self._reference_action()
                + self.residual_action_scale * policy_action,
                -1.0,
                1.0,
            )

        previous_filtered_action = self._filtered_action.copy()
        self._filtered_action += self.action_filter_coefficient * (
            control_action - self._filtered_action
        )
        action_delta = self._filtered_action - previous_filtered_action
        action_jerk = action_delta - self._previous_filtered_delta

        if not self._closing and not self._releasing:
            self._arm_target = np.clip(
                self._arm_target + self._filtered_action * self.action_scale,
                self._ctrl_low,
                self._ctrl_high,
            )

        self._update_gripper_state()
        self.data.ctrl[:4] = self._arm_target
        self.data.ctrl[4] = self._gripper_target
        for _ in range(self.frame_skip):
            if self._grasped:
                self._set_object_pose(self._attached_object_position(), self._object_yaw)
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1
        self._update_vision()
        approach_advanced = self._advance_approach_stage()
        self._try_latch_grasp()
        self._try_release_object()
        if self._grasped:
            self._set_object_pose(self._attached_object_position(), self._object_yaw)
            mujoco.mj_forward(self.model, self.data)

        info = self._get_info()
        if approach_advanced:
            self._previous_reach_distance = float(info["reach_distance"])
            self._previous_approach_joint_error = float(
                info["approach_joint_error"]
            )
        if self._episode_task == "pick":
            success_candidate = bool(
                self._grasped
                and self._phase == self.PICK_TO_STAY
                and info["stay_error"] <= self.stay_tolerance
                and info["object_clearance"] >= self.minimum_object_clearance
            )
        else:
            success_candidate = bool(
                not self._grasped
                and self._phase == self.PLACE_TO_STAY
                and info["stay_error"] <= self.stay_tolerance
                and info["placement_error"] <= self.placement_tolerance
            )
        self._success_count = self._success_count + 1 if success_candidate else 0
        is_success = self._success_count >= self.success_stable_steps
        info["success_stable_count"] = self._success_count
        info["is_success"] = is_success

        collision_failure = bool(
            info["collision_max_penetration"]
            >= self.collision_termination_depth
        )
        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        info["collision_failure"] = collision_failure
        info["nonfinite_failure"] = not finite

        reward = self._compute_reward(
            info,
            action_delta,
            action_jerk,
            policy_action,
            was_grasped,
        )
        if collision_failure:
            reward -= self.reward_weights["collision_failure"]
        if not finite:
            reward -= self.reward_weights["collision_failure"]

        terminated = bool(is_success or collision_failure or not finite)
        truncated = self._elapsed_steps >= self.max_episode_steps
        info["TimeLimit.truncated"] = bool(truncated and not terminated)

        self._previous_action = raw_action
        self._previous_filtered_delta = action_delta
        self._previous_reach_distance = float(info["reach_distance"])
        self._previous_approach_joint_error = float(
            info["approach_joint_error"]
        )
        self._previous_stay_error = float(info["stay_error"])
        self._previous_object_height = float(info["object_position"][2])
        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _sample_object_pose(
        self, options: dict[str, Any] | None
    ) -> tuple[np.ndarray, float, str]:
        if options and "object_yaw" in options:
            yaw = _wrap_to_pi(float(options["object_yaw"]))
        else:
            yaw = float(
                self.np_random.uniform(-self.object_yaw_range, self.object_yaw_range)
            )

        if options and "object_position" in options:
            position = self._vector(
                options["object_position"], "options['object_position']", 3
            )
            self._assert_supported_pose(position, yaw)
            offset = position - self._episode_object_center
            return position, yaw, self._position_bucket(offset)

        c = abs(np.cos(yaw))
        s = abs(np.sin(yaw))
        footprint = np.array(
            [
                c * self._object_half_size[0] + s * self._object_half_size[1],
                s * self._object_half_size[0] + c * self._object_half_size[1],
            ]
        )
        support_limit = self._tower_half_size[:2] - footprint - self.support_clearance
        sample_limit = np.minimum(self.object_position_range[:2], support_limit)
        if np.any(sample_limit <= 0.0):
            raise ValueError("Object footprint does not fit on the tower")

        offset_xy = self.np_random.uniform(-sample_limit, sample_limit)
        if self.np_random.random() < self.edge_sample_probability:
            axis = int(self.np_random.integers(0, 2))
            magnitude = self.np_random.uniform(
                self.edge_min_fraction * sample_limit[axis], sample_limit[axis]
            )
            offset_xy[axis] = magnitude * self.np_random.choice((-1.0, 1.0))

        z_offset = self.np_random.uniform(
            -self.object_position_range[2], self.object_position_range[2]
        )
        offset = np.array([offset_xy[0], offset_xy[1], z_offset])
        position = self._episode_object_center + offset
        self._assert_supported_pose(position, yaw)
        return position, yaw, self._position_bucket(offset)

    def _assert_supported_pose(self, position: np.ndarray, yaw: float) -> None:
        c = abs(np.cos(yaw))
        s = abs(np.sin(yaw))
        footprint = np.array(
            [
                c * self._object_half_size[0] + s * self._object_half_size[1],
                s * self._object_half_size[0] + c * self._object_half_size[1],
            ]
        )
        relative = np.abs(position[:2] - self._tower_center[:2])
        if np.any(relative + footprint + self.support_clearance > self._tower_half_size[:2] + 1e-9):
            raise ValueError("Requested object pose is not fully supported by the tower")

    def _position_bucket(self, offset: np.ndarray) -> str:
        labels = []
        for value, limit in zip(offset[:2], self.object_position_range[:2]):
            ratio = 0.0 if limit <= 1e-9 else value / limit
            labels.append("neg" if ratio < -0.34 else "pos" if ratio > 0.34 else "mid")
        return f"x_{labels[0]}__y_{labels[1]}"

    def _update_vision(self, force: bool = False) -> None:
        if not force and self._elapsed_steps % self.vision_update_interval_steps != 0:
            return
        if not force and self.np_random.random() < self.vision_dropout_probability:
            self._vision_valid = False
            self._vision_dropout_count += 1
            return

        true_position = (
            self._place_target_position
            if self._episode_task == "place" and self._phase == self.PLACE_REACH
            else self.data.site_xpos[self._object_site_id]
        )
        noise = self.np_random.normal(0.0, self.vision_position_noise_std)
        measured_position = true_position + noise
        measured_yaw = _wrap_to_pi(
            self._object_yaw
            + float(self.np_random.normal(0.0, self.vision_yaw_noise_std))
        )
        if force:
            self._vision_position = measured_position
            self._vision_yaw = measured_yaw
        else:
            alpha = self.vision_filter_coefficient
            self._vision_position += alpha * (
                measured_position - self._vision_position
            )
            yaw_delta = _wrap_to_pi(measured_yaw - self._vision_yaw)
            self._vision_yaw = _wrap_to_pi(self._vision_yaw + alpha * yaw_delta)
        self._vision_valid = True

    def _update_gripper_state(self) -> None:
        if self._phase == self.PLACE_TO_STAY:
            self._gripper_target = self.gripper_open
            return

        if self._phase == self.PLACE_REACH:
            if self._releasing:
                self._gripper_target = self.gripper_open
                self._release_steps += 1
                return
            self._gripper_target = self.gripper_close
            release_ready = self._release_ready(use_vision=True)
            self._release_gate_count = (
                self._release_gate_count + 1 if release_ready else 0
            )
            if self._release_gate_count >= self.release_stable_steps:
                self._releasing = True
                self._release_steps = 0
                self._gripper_target = self.gripper_open
                self._release_just_triggered = True
            return

        if self._grasped:
            self._gripper_target = self.gripper_close
            return

        if self._closing:
            self._gripper_target = self.gripper_close
            self._closing_steps += 1
            if self._closing_steps > self.max_closing_steps:
                self._closing = False
                self._closing_steps = 0
                self._close_gate_count = 0
                self._gripper_target = self.gripper_open
                self._failed_close = True
            return

        self._gripper_target = self.gripper_open
        open_ready = bool(
            self.data.qpos[self._gripper_qpos_addr]
            >= self.gripper_open - self.gripper_open_tolerance
        )
        close_ready = open_ready and self._close_ready(use_vision=True)
        self._close_gate_count = self._close_gate_count + 1 if close_ready else 0
        if self._close_gate_count >= self.close_stable_steps:
            self._closing = True
            self._closing_steps = 0
            self._gripper_target = self.gripper_close
            self._close_just_triggered = True

    def _try_release_object(self) -> None:
        if self._phase != self.PLACE_REACH or not self._releasing:
            return
        if self._release_steps < self.release_settle_steps:
            return
        if not self._release_ready(use_vision=False):
            return

        self._set_object_pose(self._place_target_position, self._object_yaw)
        self._grasped = False
        self._releasing = False
        self._release_just_completed = True
        self._phase = self.PLACE_TO_STAY
        self._success_count = 0
        mujoco.mj_forward(self.model, self.data)

    def _release_ready(self, use_vision: bool) -> bool:
        if self._approach_stage < len(self._episode_approach_eef_targets):
            return False
        if use_vision and not self._vision_valid:
            return False
        target_position = (
            self._vision_position if use_vision else self._place_target_position
        )
        eef_position = self.data.site_xpos[self._eef_site_id]
        delta = self._pregrasp_target(target_position) - eef_position
        target_bearing = self._bearing(target_position)
        arm_qpos = self.data.qpos[self._arm_qpos_addr]
        roll_error = abs(_wrap_to_pi(float(np.sum(arm_qpos[1:4]))))
        return bool(
            np.linalg.norm(delta) <= self.close_distance
            and np.linalg.norm(delta[:2]) <= self.close_xy_distance
            and abs(delta[2]) <= self.close_z_tolerance
            and abs(_wrap_to_pi(float(arm_qpos[0]) - target_bearing))
            <= self.close_bearing_tolerance
            and roll_error <= self.close_roll_tolerance
        )

    def _try_latch_grasp(self) -> None:
        if self._grasped or not self._closing:
            return
        if self._closing_steps < self.close_settle_steps:
            return
        if not self._close_ready(
            use_vision=False,
            distance=self.attach_distance,
            xy_distance=self.attach_xy_distance,
            z_tolerance=self.attach_z_tolerance,
        ):
            return

        eef_rotation = self.data.site_xmat[self._eef_site_id].reshape((3, 3))
        world_offset = (
            self.data.site_xpos[self._object_site_id]
            - self.data.site_xpos[self._eef_site_id]
        )
        self._grasp_attach_offset = eef_rotation.T @ world_offset
        self._grasped = True
        self._grasp_just_latched = True
        self._closing = False
        self._phase = self.PICK_TO_STAY
        self._success_count = 0

    def _close_ready(
        self,
        use_vision: bool,
        distance: float | None = None,
        xy_distance: float | None = None,
        z_tolerance: float | None = None,
    ) -> bool:
        if self._approach_stage < len(self._episode_approach_eef_targets):
            return False
        if use_vision and not self._vision_valid:
            return False
        eef_position = self.data.site_xpos[self._eef_site_id]
        object_position = (
            self._vision_position
            if use_vision
            else self.data.site_xpos[self._object_site_id]
        )
        delta = self._pregrasp_target(object_position) - eef_position
        target_bearing = self._bearing(object_position)
        arm_qpos = self.data.qpos[self._arm_qpos_addr]
        joint1 = float(arm_qpos[0])
        roll_error = abs(_wrap_to_pi(float(np.sum(arm_qpos[1:4]))))
        return bool(
            np.linalg.norm(delta) <= (self.close_distance if distance is None else distance)
            and np.linalg.norm(delta[:2])
            <= (self.close_xy_distance if xy_distance is None else xy_distance)
            and abs(delta[2])
            <= (self.close_z_tolerance if z_tolerance is None else z_tolerance)
            and abs(_wrap_to_pi(joint1 - target_bearing))
            <= self.close_bearing_tolerance
            and roll_error <= self.close_roll_tolerance
        )

    def _attached_object_position(self) -> np.ndarray:
        eef_rotation = self.data.site_xmat[self._eef_site_id].reshape((3, 3))
        return (
            self.data.site_xpos[self._eef_site_id]
            + eef_rotation @ self._grasp_attach_offset
        )

    def _set_object_pose(self, position: np.ndarray, yaw: float) -> None:
        self.data.mocap_pos[self._object_mocap_id] = position
        half_yaw = 0.5 * yaw
        self.data.mocap_quat[self._object_mocap_id] = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        )

    def _set_reach_target(self, position: np.ndarray) -> None:
        self.data.mocap_pos[self._target_mocap_id] = position
        self.data.mocap_quat[self._target_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    def _pregrasp_target(self, object_position: np.ndarray) -> np.ndarray:
        height_offset = (
            self.pregrasp_height_offset
            + self._object_nominal_half_size[2]
            - self._object_half_size[2]
        )
        return np.asarray(object_position) + np.array(
            [0.0, 0.0, height_offset]
        )

    def _configure_approach_waypoints(self, object_position: np.ndarray) -> None:
        bearing = self._bearing(object_position)
        joint_waypoints = self.approach_joint_waypoints.copy()
        if len(joint_waypoints):
            joint_waypoints[:, 0] = bearing
        self._episode_approach_joint_waypoints = joint_waypoints

        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, self._stay_key_id)
        eef_targets = []
        for joint_positions in joint_waypoints:
            scratch.qpos[self._arm_qpos_addr] = joint_positions
            scratch.ctrl[:4] = joint_positions
            mujoco.mj_forward(self.model, scratch)
            eef_targets.append(scratch.site_xpos[self._eef_site_id].copy())
        self._episode_approach_eef_targets = np.asarray(
            eef_targets, dtype=np.float64
        ).reshape((-1, 3))

    def _solve_reference_pregrasp(self, object_position: np.ndarray) -> np.ndarray:
        target = self._pregrasp_target(object_position)
        joint1 = self._bearing(target)
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, self._stay_key_id)

        def residual(joints_2_to_4: np.ndarray) -> np.ndarray:
            joints = np.concatenate(([joint1], joints_2_to_4))
            scratch.qpos[self._arm_qpos_addr] = joints
            scratch.ctrl[:4] = joints
            mujoco.mj_forward(self.model, scratch)
            position_error = 10.0 * (
                scratch.site_xpos[self._eef_site_id] - target
            )
            return np.concatenate((position_error, [np.sum(joints_2_to_4)]))

        solution = least_squares(
            residual,
            x0=self.grasped_start_joint_positions[1:],
            bounds=(self._ctrl_low[1:], self._ctrl_high[1:]),
        )
        if not solution.success:
            raise RuntimeError(f"Reference pregrasp IK failed: {solution.message}")
        return np.concatenate(([joint1], solution.x))

    def _reference_action(self) -> np.ndarray:
        if self._phase in (self.PICK_TO_STAY, self.PLACE_TO_STAY):
            joint_goal = self._stay_joint_positions
        elif self._close_gate_count > 0 or self._release_gate_count > 0:
            return np.zeros(4, dtype=np.float64)
        elif self._approach_stage < len(self._episode_approach_joint_waypoints):
            joint_goal = self._episode_approach_joint_waypoints[
                self._approach_stage
            ]
        else:
            joint_goal = self._reference_pregrasp_joint_positions
        action_limit = self.reference_action_limit
        if (
            self._phase in (self.PICK_REACH, self.PLACE_REACH)
            and self._approach_stage >= len(self._episode_approach_joint_waypoints)
        ):
            action_limit = self.final_approach_action_limit
        return np.clip(
            (joint_goal - self._arm_target) / self.action_scale,
            -action_limit,
            action_limit,
        )

    def _advance_approach_stage(self) -> bool:
        if self._phase not in (self.PICK_REACH, self.PLACE_REACH):
            return False
        if self._approach_stage >= len(self._episode_approach_joint_waypoints):
            return False
        target_joints = self._episode_approach_joint_waypoints[self._approach_stage]
        arm_qpos = self.data.qpos[self._arm_qpos_addr]
        joint_error = np.linalg.norm(arm_qpos - target_joints)
        if joint_error > self.approach_waypoint_tolerance:
            return False
        self._approach_stage += 1
        self._approach_just_advanced = True
        if (
            self._approach_stage == len(self._episode_approach_joint_waypoints)
            and self._vision_valid
        ):
            self._reference_pregrasp_joint_positions = (
                self._solve_reference_pregrasp(self._vision_position)
            )
        return True

    def _active_reach_target(self) -> np.ndarray:
        if self._phase not in (self.PICK_REACH, self.PLACE_REACH):
            return self._stay_eef_position
        if self._approach_stage < len(self._episode_approach_eef_targets):
            return self._episode_approach_eef_targets[self._approach_stage]
        return self._pregrasp_target(self._vision_position)

    def _bearing(self, position: np.ndarray) -> float:
        delta = position[:2] - self.data.xpos[self._arm_base_body_id, :2]
        return float(np.arctan2(delta[1], delta[0]))

    def _get_info(self) -> dict[str, Any]:
        eef_position = self.data.site_xpos[self._eef_site_id].copy()
        object_position = self.data.site_xpos[self._object_site_id].copy()
        active_reach_target = self._active_reach_target()
        reach_delta = active_reach_target - eef_position
        arm_qpos = self.data.qpos[self._arm_qpos_addr]
        target_bearing = self._bearing(active_reach_target)
        bearing_error = _wrap_to_pi(float(arm_qpos[0]) - target_bearing)
        roll_error = _wrap_to_pi(float(np.sum(arm_qpos[1:4])))
        stay_error = float(np.linalg.norm(arm_qpos - self._stay_joint_positions))
        collision = self._collision_metrics()
        placement_error = float(
            np.linalg.norm(object_position - self._place_target_position)
        )
        object_clearance = float(
            object_position[2]
            - self._object_half_size[2]
            - (self._tower_center[2] + self._tower_half_size[2])
        )
        approach_joint_error = 0.0
        if self._approach_stage < len(self._episode_approach_joint_waypoints):
            approach_joint_error = float(
                np.linalg.norm(
                    arm_qpos
                    - self._episode_approach_joint_waypoints[self._approach_stage]
                )
            )
        return {
            "phase": self._phase,
            "phase_name": {
                self.PICK_REACH: "PICK_REACH",
                self.PICK_TO_STAY: "PICK_TO_STAY",
                self.PLACE_REACH: "PLACE_REACH",
                self.PLACE_TO_STAY: "PLACE_TO_STAY",
            }[self._phase],
            "episode_task": self._episode_task,
            "elapsed_steps": self._elapsed_steps,
            "object_position": object_position,
            "object_initial_position": self._object_initial_position.copy(),
            "place_target_position": self._place_target_position.copy(),
            "placement_error": placement_error,
            "object_yaw": self._object_yaw,
            "object_sample_bucket": self._object_sample_bucket,
            "vision_position": self._vision_position.copy(),
            "vision_yaw": self._vision_yaw,
            "vision_valid": self._vision_valid,
            "vision_dropout_count": self._vision_dropout_count,
            "started_grasped": self._started_grasped,
            "started_released": self._started_released,
            "reference_control_enabled": self.reference_control_enabled,
            "approach_stage": self._approach_stage,
            "approach_stage_count": len(self._episode_approach_eef_targets),
            "approach_joint_error": approach_joint_error,
            "eef_position": eef_position,
            "reach_distance": float(np.linalg.norm(reach_delta)),
            "reach_xy_distance": float(np.linalg.norm(reach_delta[:2])),
            "reach_z_error": float(reach_delta[2]),
            "bearing_error": bearing_error,
            "roll_error": roll_error,
            "stay_error": stay_error,
            "object_lift": float(object_position[2] - self._object_initial_position[2]),
            "object_clearance": object_clearance,
            "gripper_position": float(self.data.qpos[self._gripper_qpos_addr]),
            "gripper_target": self._gripper_target,
            "close_ready": self._close_ready(use_vision=True),
            "close_gate_count": self._close_gate_count,
            "closing": self._closing,
            "closing_steps": self._closing_steps,
            "failed_close": self._failed_close,
            "release_ready": self._release_ready(use_vision=True),
            "release_gate_count": self._release_gate_count,
            "releasing": self._releasing,
            "release_steps": self._release_steps,
            "released": bool(
                self._episode_task == "place" and not self._grasped
            ),
            "is_grasped": self._grasped,
            "is_success": False,
            "collision_count": collision["count"],
            "collision_depth_sum": collision["depth_sum"],
            "collision_max_penetration": collision["max_penetration"],
            "domain_randomization": dict(self._episode_randomization),
        }

    def _get_observation(self) -> np.ndarray:
        arm_qpos = self.data.qpos[self._arm_qpos_addr]
        arm_qvel = self.data.qvel[self._arm_qvel_addr]
        qpos_normalized = 2.0 * (
            (arm_qpos - self._ctrl_low) / (self._ctrl_high - self._ctrl_low)
        ) - 1.0
        qvel_normalized = arm_qvel / self.joint_velocity_scale
        gripper_position = float(self.data.qpos[self._gripper_qpos_addr])
        gripper_normalized = 2.0 * (
            (gripper_position - self.gripper_close)
            / (self.gripper_open - self.gripper_close)
        ) - 1.0
        gripper_velocity = float(self.data.qvel[self._gripper_qvel_addr]) / 0.25

        eef_position = self.data.site_xpos[self._eef_site_id]
        if self._phase in (self.PICK_REACH, self.PLACE_REACH):
            active_target = self._active_reach_target()
            target_yaw = (
                self._vision_yaw
                if self._approach_stage >= len(self._episode_approach_eef_targets)
                else 0.0
            )
        else:
            active_target = self._stay_eef_position
            target_yaw = 0.0
        relative = active_target - eef_position
        position_span = self.workspace_max - self.workspace_min
        eef_normalized = 2.0 * (
            (eef_position - self.workspace_min) / position_span
        ) - 1.0
        target_normalized = 2.0 * (
            (active_target - self.workspace_min) / position_span
        ) - 1.0
        relative_normalized = 2.0 * relative / position_span
        target_bearing = self._bearing(active_target)
        roll_error = _wrap_to_pi(float(np.sum(arm_qpos[1:4]))) / np.pi

        phase_one_hot = np.zeros(4, dtype=np.float64)
        phase_one_hot[self._phase] = 1.0
        observation = np.concatenate(
            (
                qpos_normalized,
                qvel_normalized,
                [gripper_normalized],
                [gripper_velocity],
                eef_normalized,
                target_normalized,
                relative_normalized,
                [np.sin(target_bearing), np.cos(target_bearing)],
                [np.sin(target_yaw), np.cos(target_yaw)],
                [roll_error],
                [float(self._grasped)],
                phase_one_hot,
                self._previous_action,
            )
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _compute_reward(
        self,
        info: dict[str, Any],
        action_delta: np.ndarray,
        action_jerk: np.ndarray,
        policy_action: np.ndarray,
        was_grasped: bool,
    ) -> float:
        weights = self.reward_weights
        reward = -weights["time"]
        if self._phase in (self.PICK_REACH, self.PLACE_REACH):
            reach_progress = self._previous_reach_distance - info["reach_distance"]
            joint_progress = (
                self._previous_approach_joint_error
                - info["approach_joint_error"]
            )
            reward += weights["reach_progress"] * reach_progress
            reward -= weights["reach_distance"] * info["reach_distance"]
            reward += weights["approach_joint_progress"] * joint_progress
            reward -= weights["approach_joint_error"] * info["approach_joint_error"]
            reward -= weights["bearing_error"] * abs(info["bearing_error"])
            reward -= weights["roll_error"] * abs(info["roll_error"])
            if self._approach_just_advanced:
                reward += weights["approach_waypoint_bonus"]
            if self._close_just_triggered:
                reward += weights["close_ready_bonus"]
            if self._release_just_triggered:
                reward += weights["close_ready_bonus"]
            if self._phase == self.PLACE_REACH and was_grasped:
                reward += weights["holding"]
        else:
            stay_progress = self._previous_stay_error - info["stay_error"]
            reward += weights["stay_progress"] * stay_progress
            reward -= weights["stay_error"] * info["stay_error"]
            if self._phase == self.PICK_TO_STAY and self._grasped:
                lift_progress = (
                    info["object_position"][2] - self._previous_object_height
                )
                reward += weights["holding"]
                reward += weights["lift_progress"] * lift_progress
            if self._phase == self.PLACE_TO_STAY and not self._grasped:
                reward += weights["placed"]

        if self._grasp_just_latched:
            reward += weights["grasp_bonus"]
        if self._release_just_completed:
            reward += weights["release_bonus"]
        if self._failed_close:
            reward -= weights["failed_close"]
        if info["is_success"]:
            reward += weights["success"]

        reward -= weights["action_delta"] * float(np.square(action_delta).sum())
        reward -= weights["action_jerk"] * float(np.square(action_jerk).sum())
        reward -= weights["residual_action"] * float(
            np.square(policy_action).sum()
        )
        reward -= weights["joint_velocity"] * float(
            np.square(self.data.qvel[self._arm_qvel_addr]).sum()
        )
        reward -= weights["joint_limit"] * self._joint_limit_cost()
        reward -= weights["collision"] * float(info["collision_depth_sum"])
        return float(reward)

    def _joint_limit_cost(self) -> float:
        qpos = self.data.qpos[self._arm_qpos_addr]
        normalized_margin = np.minimum(
            (qpos - self._ctrl_low) / (self._ctrl_high - self._ctrl_low),
            (self._ctrl_high - qpos) / (self._ctrl_high - self._ctrl_low),
        )
        return float(np.square(np.clip(0.08 - normalized_margin, 0.0, None)).sum())

    def _collision_metrics(self) -> dict[str, float | int]:
        count = 0
        depth_sum = 0.0
        max_penetration = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            protected_pair = (
                geom1 in self._body_collision_geom_ids
                and geom2 in self._arm_collision_geom_ids
            ) or (
                geom2 in self._body_collision_geom_ids
                and geom1 in self._arm_collision_geom_ids
            )
            if not protected_pair:
                continue
            penetration = max(0.0, -float(contact.dist))
            count += 1
            depth_sum += penetration
            max_penetration = max(max_penetration, penetration)
        return {
            "count": count,
            "depth_sum": depth_sum,
            "max_penetration": max_penetration,
        }

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing MuJoCo object: {name}")
        return int(object_id)

    def _named_ids(
        self, object_type: mujoco.mjtObj, names: tuple[str, ...]
    ) -> np.ndarray:
        return np.array(
            [self._name_id(object_type, name) for name in names],
            dtype=np.int32,
        )
