import mujoco

model_path = "assets/mjcf/turtlebot3_manipulator/scene.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("Model loaded:", model_path)
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("ngeom:", model.ngeom)

for _ in range(1000):
    mujoco.mj_step(model, data)

print("Simulation step OK")
print("qpos:", data.qpos[:])
