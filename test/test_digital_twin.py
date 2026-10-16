"""Cross-check native twin kinematics and ensure styling preserves delivery physics."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from view_twin import load_scene, update_shelf
from twin_style import apply_style


def transform(xyz, rpy):
    r, p, y = rpy
    rx = np.array([[1,0,0], [0,np.cos(r),-np.sin(r)], [0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)], [0,1,0], [-np.sin(p),0,np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0], [np.sin(y),np.cos(y),0], [0,0,1]])
    result = np.eye(4)
    result[:3,:3], result[:3,3] = rz @ ry @ rx, xyz
    return result


@pytest.mark.parametrize("values", [[.104311,.027612,-.001534,-1.638291,.019],
                                  [-.3,-.5,.6,-.2,.008], [.8,.4,-.2,.3,-.005]])
def test_native_bodies_match_source_urdf(values):
    spec = json.loads((ROOT / "assets/twin/source.json").read_text())
    m, d = load_scene("shelf")
    d.ctrl[:] = values
    update_shelf(m, d)
    controls = dict(zip(["joint1","joint2","joint3","joint4","gripper_left_joint"], values))
    controls["gripper_right_joint"] = values[-1]
    world = {name: np.eye(4) for name in spec["roots"]}
    pending = list(spec["joints"])
    while pending:
        before = len(pending)
        for j in pending[:]:
            if j["parent"] not in world:
                continue
            motion = np.eye(4)
            value = controls.get(j["name"], 0)
            if j["type"] == "prismatic":
                motion[:3,3] = np.array(j["axis"]) * value
            elif j["type"] in ("revolute", "continuous"):
                axis = np.array(j["axis"], dtype=float)
                axis /= np.linalg.norm(axis)
                x,y,z = axis
                skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                motion[:3,:3] = np.eye(3) + np.sin(value)*skew + (1-np.cos(value))*(skew@skew)
            world[j["child"]] = world[j["parent"]] @ transform(j["xyz"],j["rpy"]) @ motion
            pending.remove(j)
        assert len(pending) < before, "Disconnected or cyclic source hierarchy"
    for name, expected in world.items():
        body = m.body(name).id
        np.testing.assert_allclose(d.xpos[body], expected[:3,3], atol=1e-10)
        np.testing.assert_allclose(d.xmat[body].reshape(3,3), expected[:3,:3], atol=1e-10)


def test_shelf_controls_clamp_mimic_and_reset():
    m,d = load_scene("shelf")
    d.ctrl[:] = 100
    update_shelf(m,d)
    for i in range(m.nu):
        j = m.actuator_trnid[i,0]
        assert d.qpos[m.jnt_qposadr[j]] == m.actuator_ctrlrange[i,1]
    assert d.qpos[m.joint("gripper_left_joint").qposadr[0]] == d.qpos[m.joint("gripper_right_joint").qposadr[0]]
    mujoco.mj_resetDataKeyframe(m,d,0)
    update_shelf(m,d)
    assert d.ctrl[0] == pytest.approx(.104311)
    assert not np.any(m.geom_contype) and not np.any(m.geom_conaffinity)


def test_style_preserves_legacy_delivery_dynamics():
    path = str(ROOT / "assets/mjcf/turtlebot3_manipulator/scene.xml")
    original, styled = mujoco.MjModel.from_xml_path(path), mujoco.MjModel.from_xml_path(path)
    apply_style(styled)
    a,b = mujoco.MjData(original), mujoco.MjData(styled)
    for m,d in [(original,a), (styled,b)]:
        mujoco.mj_resetDataKeyframe(m,d,0)
        for _ in range(1000):
            mujoco.mj_step(m,d)
    assert (styled.nq,styled.nv,styled.nu) == (8,8,5)
    np.testing.assert_array_equal(a.qpos,b.qpos)
    np.testing.assert_array_equal(a.qvel,b.qvel)
    np.testing.assert_array_equal(a.qacc,b.qacc)
