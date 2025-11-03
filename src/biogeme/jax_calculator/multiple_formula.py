"""
This module defines the MultiRowEvaluator class, which evaluates multiple expressions
on a given database using JAX for efficient batched computation. It returns results
as a pandas DataFrame with one column per expression and one row per observation.

Michel Bierlaire
Wed Apr 2 13:10:17 2025
"""

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp

from biogeme.exceptions import BiogemeError
from biogeme.expressions import build_vectorized_function
from biogeme.floating_point import JAX_FLOAT, NUMPY_FLOAT
from biogeme.model_elements import ModelElements


class MultiRowEvaluator:
    """
    Evaluates multiple expressions on a common dataset using JAX and returns results
    as a pandas DataFrame. This class compiles all expressions into JAX functions and
    evaluates them efficiently in a single batched operation.

    :param model_elements: Object containing the expressions and all elements needed to calculate them.
    """

    def __init__(
        self,
        model_elements: ModelElements,
        numerically_safe: bool,
        use_jit: bool,
    ):
        if model_elements is None:
            raise BiogemeError('A model must be provided.')
        self.multiple_model_elements = model_elements
        self.free_betas_names = model_elements.expressions_registry.free_betas_names
        self.data_jax = model_elements.database.data_jax
        self.draws_jax = model_elements.draws_management.draws_jax
        self.names = list(model_elements.expressions.keys())
        n_rv = (
            self.multiple_model_elements.expressions_registry.number_of_random_variables
        )
        self.random_variables_jax = jnp.zeros((n_rv,), dtype=JAX_FLOAT)

        self.vectorized_functions = [
            build_vectorized_function(
                expr.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                ),
                use_jit=use_jit,
            )
            for expr in self.multiple_model_elements.expressions.values()
        ]

        def evaluate_all_impl(params, data, draws, rv):
            return jnp.stack(
                [vf(params, data, draws, rv) for vf in self.vectorized_functions],
                axis=1,
            )

        if use_jit:
            self._evaluate_all = jax.jit(evaluate_all_impl)
        else:
            self._evaluate_all = evaluate_all_impl

    def evaluate(self, the_betas: dict[str, float]) -> pd.DataFrame:
        """
        Evaluates all expressions using the provided beta values.

        :param the_betas: A dictionary mapping beta names to their numerical values.
        :return: A pandas DataFrame with one column per expression and one row per observation.
        """
        param_vector = (
            self.multiple_model_elements.expressions_registry.get_betas_array(the_betas)
        )
        values = self._evaluate_all(
            param_vector, self.data_jax, self.draws_jax, self.random_variables_jax
        )
        return pd.DataFrame(np.asarray(values, dtype=NUMPY_FLOAT), columns=self.names)
