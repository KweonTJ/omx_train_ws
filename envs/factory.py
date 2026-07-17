"""Configuration loading and environment construction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .arm_randomized_grasp_env import ArmRandomizedGraspEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/arm_grasp_randomized_ppo.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def environment_kwargs(
    config: dict[str, Any], stage_name: str | None = None
) -> dict[str, Any]:
    kwargs = deepcopy(config["environment"])
    if stage_name is not None:
        stages = {stage["name"]: stage for stage in config["curriculum"]["stages"]}
        if stage_name not in stages:
            raise ValueError(f"Unknown curriculum stage: {stage_name}")
        kwargs = _deep_update(kwargs, stages[stage_name].get("environment", {}))

    model_path = Path(config["model"]["path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    kwargs["model_path"] = str(model_path.resolve())
    return kwargs


def make_grasp_env(
    config: dict[str, Any],
    stage_name: str | None = None,
    render_mode: str | None = None,
) -> ArmRandomizedGraspEnv:
    kwargs = environment_kwargs(config, stage_name)
    kwargs["render_mode"] = render_mode
    return ArmRandomizedGraspEnv(**kwargs)
