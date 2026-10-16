"""Build a native display MJCF from the exact web twin snapshot (no ROS needed).

All robot/shelf geoms are decorative. This is a kinematic inspection scene,
not a collision model or a replacement for the trained delivery environment.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "assets/twin"
ARM = ["joint1", "joint2", "joint3", "joint4", "gripper_left_joint"]


def fmt(values):
    return " ".join(f"{v:.12g}" for v in values)


def quaternion(rpy):
    # URDF uses fixed-axis XYZ (Rz @ Ry @ Rx). MJCF expects w,x,y,z.
    r, p, y = [v / 2 for v in rpy]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return fmt([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def build():
    spec = json.loads((DIRECTORY / "source.json").read_text())
    for relative, expected in spec["assets"].items():
        path = (DIRECTORY / relative).resolve()
        if not path.is_relative_to(DIRECTORY.resolve()):
            raise ValueError("Asset outside twin directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset checksum mismatch: {relative}")
    root = ET.Element("mujoco", model="OMX shelf twin — kinematic display")
    ET.SubElement(root, "compiler", angle="radian", eulerseq="xyz", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="0.002")
    ET.SubElement(root, "statistic", center="0.12 0 0.22", extent="0.8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="45", elevation="-25", offwidth="1200", offheight="800")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.035 0.055 0.09", rgb2="0.035 0.055 0.09", width="64", height="64")
    ET.SubElement(asset, "material", name="floor_mat", rgba="0.055 0.08 0.125 1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.3 -0.7 1.5", dir="-0.2 0.5 -1", diffuse="0.8 0.85 0.95")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .02", material="floor_mat", contype="0", conaffinity="0")
    links = {link["name"]: link for link in spec["links"]}
    children = {}
    for joint in spec["joints"]:
        children.setdefault(joint["parent"], []).append(joint)
    meshes, materials, textures, order = {}, {}, {}, []

    def add_link(name, parent, joint=None):
        attributes = {"name": name}
        if joint:
            attributes.update(pos=fmt(joint["xyz"]), quat=quaternion(joint["rpy"]))
        body = ET.SubElement(parent, "body", **attributes)
        if joint and joint["type"] in ("revolute", "prismatic", "continuous"):
            attrs = dict(name=joint["name"], axis=fmt(joint["axis"]),
                         type="slide" if joint["type"] == "prismatic" else "hinge")
            if "limit" in joint and "lower" in joint["limit"]:
                attrs["range"] = fmt([joint["limit"]["lower"], joint["limit"]["upper"]])
            ET.SubElement(body, "joint", **attrs)
            # Only mj_forward is used. These are display inertias, not identified dynamics.
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1", diaginertia=".0001 .0001 .0001")
            order.append(joint["name"])
        for i, v in enumerate(links[name]["visuals"]):
            shape, mat = v["geometry"], v["material"]
            material_key = json.dumps(mat, sort_keys=True)
            if material_key not in materials:
                mat_name = f"material_{len(materials)}"
                attrs = dict(name=mat_name, rgba=fmt(mat["rgba"]), shininess=".15", specular=".2")
                if mat.get("texture"):
                    filename = mat["texture"]
                    if filename not in textures:
                        textures[filename] = f"texture_{len(textures)}"
                        ET.SubElement(asset, "texture", name=textures[filename], type="2d", file=filename)
                    attrs["texture"] = textures[filename]
                ET.SubElement(asset, "material", **attrs)
                materials[material_key] = mat_name
            attrs = dict(name=f"{name}_visual_{i}", pos=fmt(v["xyz"]), quat=quaternion(v["rpy"]),
                         type=shape["type"], material=materials[material_key],
                         contype="0", conaffinity="0", mass="0", group="1")
            if shape["type"] == "mesh":
                key = (shape["file"], tuple(shape["scale"]))
                if key not in meshes:
                    meshes[key] = f"mesh_{len(meshes)}"
                    ET.SubElement(asset, "mesh", name=meshes[key], file=shape["file"], scale=fmt(shape["scale"]))
                attrs["mesh"] = meshes[key]
            elif shape["type"] == "box":
                attrs["size"] = fmt([v / 2 for v in shape["size"]])
            elif shape["type"] == "cylinder":
                attrs["size"] = fmt([shape["radius"], shape["length"] / 2])
            else:
                attrs["size"] = str(shape["radius"])
            ET.SubElement(body, "geom", **attrs)
        for child in children.get(name, []):
            add_link(child["child"], body, child)

    for name in spec["roots"]:
        add_link(name, world)
    # Same four colored, ArUco-marked objects and placement as the browser.
    colors = [(.886, .631, .498), (.541, .745, .702), (.655, .631, .796), (.843, .745, .51)]
    for i in range(4):
        x, y = .3 - .06, (-.08 if i % 2 == 0 else .08)
        z = .1 + (i // 2) * .156 + .028 + .025
        body = ET.SubElement(world, "body", name=f"object_{i+1}", mocap="true", pos=fmt([x, y, z]))
        ET.SubElement(body, "geom", type="box", size=".0275 .0275 .025",
                      rgba=fmt([*colors[i], 1]), contype="0", conaffinity="0")
        ET.SubElement(body, "geom", type="box", size=".0003 .0266665 .0266665", pos="-.0278 0 0",
                      rgba="1 1 1 1", contype="0", conaffinity="0")
        for row, bits in enumerate(spec["markers"][str(i+1)]):
            for col, black in enumerate(bits):
                if black:
                    ET.SubElement(body, "geom", type="box", size=fmt([.0002, .04/12, .04/12]),
                                  pos=fmt([-.0282, .02-(col+.5)*.04/6, .02-(row+.5)*.04/6]),
                                  rgba=".01 .01 .01 1", contype="0", conaffinity="0")
    actuator = ET.SubElement(root, "actuator")
    limits = {j["name"]: j["limit"] for j in spec["joints"] if "limit" in j}
    for name in ARM:
        ET.SubElement(actuator, "position", name=name, joint=name, kp="1",
                      ctrlrange=fmt([limits[name]["lower"], limits[name]["upper"]]))
    keyframes = ET.SubElement(root, "keyframe")
    values = {**spec["initialJoints"], "gripper_right_joint": spec["initialJoints"]["gripper_left_joint"]}
    ET.SubElement(keyframes, "key", name="stay", qpos=fmt([values.get(n, 0) for n in order]),
                  ctrl=fmt([values[n] for n in ARM]))
    ET.indent(root)
    ET.ElementTree(root).write(DIRECTORY / "shelf.xml", encoding="unicode")
    print(f"Exported {len(links)} links, {len(meshes)} meshes, {len(order)} display joints")


if __name__ == "__main__":
    build()
