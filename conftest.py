import os

# Emulate 8 CPU devices so the multi-device sharding tests run on a CPU-only
# host. Must be set before jax is imported; pytest loads conftest first. Append
# rather than overwrite: TPU/GPU VMs often ship with XLA_FLAGS already set.
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    _count = "--xla_force_host_platform_device_count=8"
    os.environ["XLA_FLAGS"] = f"{_flags} {_count}".strip()
