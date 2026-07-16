import jax
import mujoco

print("JAX devices:")
for d in jax.devices():
    print(" -", d)

print("JAX backend:", jax.default_backend())
print("MuJoCo version:", mujoco.__version__)

try:
    from mujoco import mjx
    print("MJX import: OK")
except Exception as e:
    print("MJX import failed:", e)
