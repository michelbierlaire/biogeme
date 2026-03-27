from __future__ import annotations

import os
from typing import Any

import jax


def report_jax_environment() -> dict[str, Any]:
    """Return a snapshot of the JAX runtime and thread-related environment."""
    return {
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": jax.config.read("jax_enable_x64"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }
