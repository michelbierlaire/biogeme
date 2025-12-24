"""Arithmetic expressions accepted by Biogeme:exp

Michel Bierlaire
10.04.2025 11:48
"""

from __future__ import annotations

import logging

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from jax import numpy as jnp

from . import (
    ExpressionOrNumeric,
)
from .bayesian import Dimension, PymcModelBuilderType
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class DistributedParameter(UnaryOperator):
    """
    Distributed parameter in Bayesian estimation
    """

    def __init__(
        self,
        name: str,
        child: ExpressionOrNumeric,
    ) -> None:
        """Constructor

        :param child: expression of the parameter. Most of the time, mu + sigma * xi.
        """
        super().__init__(child)
        self.name = name
        self.panel_column = None

    def deep_flat_copy(self) -> DistributedParameter:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(name=self.name, child=copy_child)

    def __str__(self) -> str:
        return f'DistributedParameter({self.name}, {self.child})'

    def __repr__(self) -> str:
        return f'DistributedParameter({self.name}, {self.child}, {repr(self.child)})'

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        child_jax = self.child.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return child_value

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        child_pymc = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            model = pm.modelcontext(None)  # Get current active model context
            if self.name in model.named_vars:
                return model.named_vars[self.name]
            child_value = child_pymc(dataframe=dataframe)
            # Panel case: map individuals -> observations using the panel column.
            if self.panel_column is not None and self.panel_column in dataframe.columns:
                # Panel ids for each observation; we map them to integer indices.
                # Assumption: the order of individuals used to build the draws is
                # consistent with the codes produced here (e.g., via factorize).
                panel_ids = dataframe[self.panel_column].to_numpy()

                # Map arbitrary ids to 0..(n_individuals-1).
                codes, uniques = pd.factorize(panel_ids, sort=True)
                # We rely on child_value having shape (n_individuals, ...) with the same
                # ordering as `uniques`. If not, the calling code must ensure consistency.
                idx = pt.as_tensor_variable(codes, dtype="int64")

                indiv_name = f"{self.name}_per_individual"
                if indiv_name not in model.named_vars:
                    pm.Deterministic(
                        indiv_name,
                        child_value,
                        dims=(Dimension.INDIVIDUALS.value,),
                    )

                # Broadcast from individual-level to observation-level.
                child_value_obs = child_value[idx]

                return pm.Deterministic(
                    self.name,
                    child_value_obs,
                    dims=(Dimension.OBS.value,),
                )

            return pm.Deterministic(self.name, child_value, dims=(Dimension.OBS.value,))

        return builder
