"""Viewer-only palette shared by native MuJoCo and the portfolio twin.

Never changes collisions, geometry, masses, joints, controls or training files.
"""
import mujoco
import numpy as np


def apply_style(model):
    palette = {
        "floor_mat": [.055, .080, .125, 1],
        "dark_mat": [.085, .11, .16, 1],
        "base_mat": [.34, .39, .46, 1],
        "camera_mat": [.46, .51, .58, 1],
        "tower_mat": [.42, .49, .56, 1],
        "object_mat": [.89, .63, .50, 1],
    }
    for name, color in palette.items():
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if idx >= 0:
            model.mat_rgba[idx] = color
    model.vis.headlight.ambient[:] = [.34, .37, .43]
    model.vis.headlight.diffuse[:] = [.72, .76, .84]
    model.vis.headlight.specular[:] = [.12, .12, .12]
    model.vis.rgba.haze[:] = [.035, .055, .09, 1]
    model.vis.quality.shadowsize = 2048


def add_grid(scene):
    """Non-physical viewer overlay, independent of the MjModel."""
    for axis in range(2):
        for value in np.linspace(-1, 1, 21):
            if scene.ngeom >= scene.maxgeom:
                return
            start = [-1., value, .0015]
            end = [1., value, .0015]
            if axis:
                start, end = [value, -1., .0015], [value, 1., .0015]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([.20, .27, .36, .45], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1, start, end)
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1


def set_camera(camera):
    camera.lookat[:] = [.12, 0, .22]
    camera.distance = 1.25
    camera.azimuth = 45
    camera.elevation = -25
