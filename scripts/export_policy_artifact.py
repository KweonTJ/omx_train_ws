"""Build reproducible training and ROS handoff artifacts for a PPO policy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MODEL_MANIFEST = (
    ROOT / "ros_export/arm_grasp_pitch_hold_v3/source_model_manifest.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _validate_metadata(metadata: dict[str, Any], version: str) -> None:
    if metadata.get("policy_version") != version:
        raise ValueError(
            f"policy_version mismatch: {metadata.get('policy_version')} != {version}"
        )
    if metadata.get("supported_commands") != ["PICK", "PLACE"]:
        raise ValueError("Export requires supported_commands: [PICK, PLACE]")
    if not metadata.get("delivery_sequence_trained"):
        raise ValueError("Export requires delivery_sequence_trained: true")
    if metadata.get("observation_size") != 33 or metadata.get("action_size") != 4:
        raise ValueError("Export requires the deployed 33D observation / 4D action")


def _copy_evaluations(source_dir: Path, stage_dir: Path, version: str) -> None:
    evaluation_source = source_dir / "evaluation"
    if not evaluation_source.is_dir():
        raise FileNotFoundError(f"Missing evaluation directory: {evaluation_source}")
    evaluation_target = stage_dir / "evaluation"
    evaluation_target.mkdir()
    reports = sorted(evaluation_source.glob("*.yaml"))
    if not reports:
        raise FileNotFoundError(f"No evaluation reports: {evaluation_source}")
    for report_path in reports:
        report = _load_yaml(report_path)
        report["policy"] = f"policies/exported/{version}/policy.zip"
        _write_yaml(evaluation_target / report_path.name, report)


def _runtime_manifest(
    version: str,
    metadata: dict[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    return {
        "manifest_version": 2,
        "policy_version": version,
        "policy_file": "policy.zip",
        "source": {
            "workspace": str(ROOT),
            "relative_path": str(source_dir.relative_to(ROOT) / "arm_grasp_latest.zip"),
            "completed_stages": metadata["completed_stages"],
            "selected_counter": int(metadata["total_timesteps"]),
        },
        "contract": {
            "observation_schema_version": int(metadata["observation_schema_version"]),
            "observation_size": int(metadata["observation_size"]),
            "action_schema_version": int(metadata["action_schema_version"]),
            "action_size": int(metadata["action_size"]),
            "joint_names": metadata["joint_names"],
            "supported_commands": metadata["supported_commands"],
            "delivery_sequence_trained": True,
            "control_mode": metadata["control_mode"],
            "gripper_pitch_hold_enabled": bool(
                metadata["gripper_pitch_hold_enabled"]
            ),
            "gripper_pitch_target": float(metadata["gripper_pitch_target"]),
            "gripper_pitch_joint4_max_step": float(
                metadata["gripper_pitch_joint4_max_step"]
            ),
            "transport_controller": "external_mobile_base",
            "transport_arm_state": "PICK_TO_STAY_HOLDING_OBJECT",
            "training_frame": "base_footprint",
            "ros_target_frame": "base_link",
            "ros_to_training_offset_xyz": [0.0, 0.0, 0.016],
        },
    }


def _write_checksums(stage_dir: Path) -> None:
    files = sorted(
        path for path in stage_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(stage_dir)}"
        for path in files
    ]
    (stage_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_staging(
    source_dir: Path,
    version: str,
    source_model_manifest: Path,
    staging_dir: Path,
) -> None:
    metadata_path = source_dir / "policy_metadata.yaml"
    metadata = _load_yaml(metadata_path)
    _validate_metadata(metadata, version)

    shutil.copy2(source_dir / "arm_grasp_latest.zip", staging_dir / "policy.zip")
    shutil.copy2(metadata_path, staging_dir / "policy_metadata.yaml")
    shutil.copy2(source_dir / "training_config.yaml", staging_dir / "training_config.yaml")
    shutil.copy2(source_model_manifest, staging_dir / "source_model_manifest.yaml")
    _copy_evaluations(source_dir, staging_dir, version)
    _write_yaml(
        staging_dir / "runtime_manifest.yaml",
        _runtime_manifest(version, metadata, source_dir),
    )
    _write_checksums(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--source-model-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MODEL_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    source_model_manifest = args.source_model_manifest.expanduser().resolve()
    destinations = (
        ROOT / "policies/exported" / args.version,
        ROOT / "ros_export" / args.version,
    )
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing artifact: {destination}")

    with tempfile.TemporaryDirectory(prefix=f"{args.version}-", dir="/tmp") as temporary:
        staging_dir = Path(temporary) / args.version
        staging_dir.mkdir()
        _build_staging(
            source_dir,
            args.version,
            source_model_manifest,
            staging_dir,
        )
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging_dir, destination)
            print(f"Exported: {destination}")
            print(f"Policy SHA-256: {_sha256(destination / 'policy.zip')}")


if __name__ == "__main__":
    main()
