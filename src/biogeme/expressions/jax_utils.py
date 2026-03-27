"""Define various items for Jax

Michel Bierlaire
Tue Mar 18 18:28:07 2025
"""

from collections.abc import Callable
from typing import TypeAlias

import jax.numpy as jnp
from biogeme.profiling import JaxExecutionProfile
from jax import jit, vmap

JaxFunctionType: TypeAlias = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]

_VECTOR_FUNCTION_CACHE: dict[tuple[int, bool], JaxFunctionType] = {}


def build_vectorized_function(
    the_function: JaxFunctionType,
    use_jit: bool,
    profiler: JaxExecutionProfile | None = None,
    profile_name: str | None = None,
) -> JaxFunctionType:
    """Build and cache the row-wise vectorized version of a JAX function.

    The returned callable applies ``the_function`` observation by observation,
    vectorizing jointly over the rows of ``data`` and ``draws`` while
    broadcasting ``parameters`` and ``random_variables``.
    """
    profiler = profiler if profiler is not None else JaxExecutionProfile()
    function_name = (
        profile_name
        if profile_name is not None
        else f'vectorized_function_{id(the_function)}'
    )
    cache_key = (id(the_function), use_jit, id(profiler), function_name)
    cached = _VECTOR_FUNCTION_CACHE.get(cache_key)
    if cached is not None:
        profiler.add_note(f'Cache hit for {function_name}.')
        return cached

    def row_function(row, draw, parameters, random_variables):
        return the_function(parameters, row, draw, random_variables)

    mapped_function = vmap(
        row_function,
        in_axes=(0, 0, None, None),
    )

    if use_jit:
        compiled_function = jit(mapped_function)

        def vectorized_function(parameters, data, draws, random_variables):
            return profiler.timed_call(
                function_name,
                compiled_function,
                data,
                draws,
                parameters,
                random_variables,
            )

    else:

        def vectorized_function(parameters, data, draws, random_variables):
            return profiler.timed_call(
                function_name,
                mapped_function,
                data,
                draws,
                parameters,
                random_variables,
            )

    profiler.record_build(function_name)
    _VECTOR_FUNCTION_CACHE[cache_key] = vectorized_function
    return vectorized_function
