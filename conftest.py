import os

# Emulate 8 CPU devices so the multi-device sharding tests run on any host.
# Set before jax is imported; pytest loads conftest before any test module.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
