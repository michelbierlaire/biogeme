"""Define various items for Jax

Michel Bierlaire
Tue Mar 18 18:28:07 2025
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax import jit, vmap

JaxFunctionType = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.array
]


def build_vectorized_function(the_function, use_jit: bool):
    """Build the function that is applied to each row of the databaser"""

    def vectorized_function(parameters, data, draws, random_variables):
        return vmap(
            lambda row, draw: the_function(parameters, row, draw, random_variables),
            in_axes=(0, 0),
        )(data, draws)

    return jit(vectorized_function) if use_jit else vectorized_function
