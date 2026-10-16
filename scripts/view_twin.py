"""Open the shelf kinematic twin or the original delivery physics sandbox.

Use MuJoCo's right-hand Control panel for joint targets; R resets the scene.
Shelf uses mj_forward only. Delivery uses the original physics and controls.
Neither mode executes a learned policy; view_grasp_policy.py does that.
"""
import argparse
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from twin_style import add_grid, apply_style, set_camera

ROOT = Path(__file__).resolve().parents[1]


def load_scene(name):
    path = ROOT / ("assets/twin/shelf.xml" if name == "shelf" else
                   "assets/mjcf/turtlebot3_manipulator/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    apply_style(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def update_shelf(model, data):
    # Absolute joint targets from the built-in Control panel; mimic the two fingers.
    for i in range(model.nu):
        joint = model.actuator_trnid[i, 0]
        data.qpos[model.jnt_qposadr[joint]] = np.clip(data.ctrl[i], *model.actuator_ctrlrange[i])
    left = model.joint("gripper_left_joint").qposadr[0]
    right = model.joint("gripper_right_joint").qposadr[0]
    data.qpos[right] = data.qpos[left]
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["shelf", "delivery"], default="shelf")
    args = parser.parse_args()
    model, data = load_scene(args.scene)
    reset = threading.Event()
    def on_key(key):
        if key in (ord("R"), ord("r")):
            reset.set()
    print(f"OMX / {args.scene.upper()} — " + ("KINEMATIC DISPLAY" if args.scene == "shelf" else "DELIVERY PHYSICS SANDBOX"))
    print("Right panel > Control: joint targets. R: reset. Drag: orbit. Scroll: zoom.")
    print("No policy inference or hardware commands are executed.")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=False, show_right_ui=True) as viewer:
        set_camera(viewer.cam)
        with viewer.lock():
            add_grid(viewer.user_scn)
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if reset.is_set():
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    reset.clear()
                if args.scene == "shelf":
                    update_shelf(model, data)
                else:
                    for _ in range(8):
                        mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 1/60 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
