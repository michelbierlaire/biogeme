"""Sets the type of floating point used by Biogeme

Michel Bierlaire
Mon Mar 31 09:57:37 2025
"""

import logging
import os

import numpy as np
from jax import config, numpy as jnp

from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)

raw_value = os.getenv('BIOGEME_FLOAT_TYPE', '64')
if raw_value not in {'32', '64'}:
    logger.warning(
        f'Invalid float type: {raw_value}. Valid values: "32" or "64". Update the environment variable BIOGEME_FLOAT_TYPE. "64" is used by default'
    )
    raw_value = '64'
FLOAT_TYPE = int(raw_value)

if FLOAT_TYPE == 64:
    config.update("jax_enable_x64", True)

if FLOAT_TYPE != 64 and FLOAT_TYPE != 32:
    raise BiogemeError('FLOAT_TYPE must be 32 or 64')

FLOAT = 'float64' if FLOAT_TYPE == 64 else 'float32'
NUMPY_FLOAT = np.float64 if FLOAT_TYPE == 64 else np.float32
PANDAS_FLOAT = FLOAT
JAX_FLOAT = jnp.dtype(jnp.float32) if FLOAT_TYPE == 32 else jnp.dtype(jnp.float64)
EPSILON = jnp.finfo(JAX_FLOAT).eps
SQRT_EPS = jnp.sqrt(EPSILON)
SMALL_POSITIVE = 10 * EPSILON
LOG_CLIP_MIN = SQRT_EPS
MAX_EXP_ARG = jnp.log(jnp.finfo(JAX_FLOAT).max)
MIN_EXP_ARG = jnp.log(jnp.finfo(JAX_FLOAT).tiny)
MOST_NEGATIVE = jnp.finfo(JAX_FLOAT).min
