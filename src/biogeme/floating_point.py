"""Sets the type of floating point used by Biogeme

Michel Bierlaire
Mon Mar 31 09:57:37 2025
"""

import jax.numpy as jnp
import numpy as np
from jax import config

from biogeme.exceptions import BiogemeError

FLOAT_TYPE = 32

if FLOAT_TYPE == 64:
    config.update("jax_enable_x64", True)

if FLOAT_TYPE != 64 and FLOAT_TYPE != 32:
    raise BiogemeError('FLOAT_TYPE must bw 32 or 64')

FLOAT = 'float64' if FLOAT_TYPE == 64 else 'float32'
NUMPY_FLOAT = np.float64 if FLOAT_TYPE == 64 else np.float32
PANDAS_FLOAT = FLOAT
JAX_FLOAT = jnp.dtype(jnp.float32) if FLOAT_TYPE == 32 else jnp.dtype(jnp.float64)
SQRT_EPS = jnp.sqrt(jnp.finfo(JAX_FLOAT).eps)
LOG_CLIP_MIN = SQRT_EPS
MAX_EXP_ARG = jnp.log(jnp.finfo(JAX_FLOAT).max)
MIN_EXP_ARG = jnp.log(jnp.finfo(JAX_FLOAT).tiny)
