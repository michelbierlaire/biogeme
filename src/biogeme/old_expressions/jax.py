"""Define various items for Jax

Michel Bierlaire
Tue Mar 18 18:28:07 2025
"""

from collections.abc import Callable
import jax.numpy as jnp
from jax import jit, vmap

JaxFunctionType = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
JAX_FLOAT = jnp.dtype(jnp.float32)


def build_vectorized_function(the_function):
    """Build the function that is applied to each row of the databaser"""

    @jit
    def vectorized_function(parameters, data, draws):
        return vmap(
            lambda row, draw: the_function(parameters, row, draw), in_axes=(0, 0)
        )(data, draws)

    return vectorized_function
