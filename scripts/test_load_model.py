import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/mjcf/turtlebot3_manipulator/scene.xml"
MANIFEST_PATH = MODEL_PATH.with_name("model_manifest.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(f"Missing MuJoCo object: {name}")
    return object_id


manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
assert manifest["model"]["file"] == MODEL_PATH.name
assert sha256(MODEL_PATH) == manifest["model"]["sha256"]

mesh_dir = MODEL_PATH.parent / "meshes/open_manipulator_x"
for mesh_name, expected_hash in manifest["meshes"].items():
    mesh_path = mesh_dir / mesh_name
    assert sha256(mesh_path) == expected_hash, f"Mesh hash mismatch: {mesh_path}"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

assert (model.nq, model.nv, model.nu) == (8, 8, 5)

tower_id = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "task_tower")
object_id = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_object")
object_body_id = name_id(
    model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object_body"
)
place_site_id = name_id(
    model, mujoco.mjtObj.mjOBJ_SITE, "handoff_place_site"
)
stay_key_id = name_id(model, mujoco.mjtObj.mjOBJ_KEY, "stay")

tower_size = 2.0 * model.geom_size[tower_id]
object_size = 2.0 * model.geom_size[object_id]
object_center = model.body_pos[object_body_id]
tower_top = model.geom_pos[tower_id, 2] + model.geom_size[tower_id, 2]
object_bottom = object_center[2] - model.geom_size[object_id, 2]

np.testing.assert_allclose(tower_size, [0.13, 0.13, 0.17])
np.testing.assert_allclose(object_size, [0.06, 0.055, 0.055])
np.testing.assert_allclose(object_bottom, tower_top)
np.testing.assert_allclose(model.site_pos[place_site_id], object_center)

mujoco.mj_resetDataKeyframe(model, data, stay_key_id)
mujoco.mj_forward(model, data)
for _ in range(1000):
    mujoco.mj_step(model, data)

assert np.isfinite(data.qpos).all()
assert np.isfinite(data.qvel).all()

print(f"Model loaded: {MODEL_PATH}")
print("Model and mesh manifest OK")
print(f"nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
print(f"tower size [m]: {tower_size}")
print(f"object size [m]: {object_size}")
print(f"tower top/object bottom [m]: {tower_top}/{object_bottom}")
print("Simulation step OK")
