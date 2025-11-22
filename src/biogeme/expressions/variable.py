"""Arithmetic expressions accepted by Biogeme: variables

Michel Bierlaire
Fri Jun 27 2025, 14:43:42
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pandas as pd
import pymc as pm
from pytensor.tensor import TensorVariable

from biogeme.bayesian_estimation import check_shape
from biogeme.exceptions import BiogemeError
from .bayesian import Dimension, PymcModelBuilderType
from .elementary_expressions import Elementary
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class Variable(Elementary):
    """Explanatory variable

    This represents the explanatory variables of the choice
    model. Typically, they come from the data set.
    """

    expression_type = TypeOfElementaryExpression.VARIABLE

    def __init__(self, name: str):
        """Constructor

        :param name: name of the variable.
        :type name: string
        """
        super().__init__(name)

    def deep_flat_copy(self) -> Variable:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(name=self.name)

    @property
    def safe_variable_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for variable {self.name}")
        return self.specific_id

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.array:
            return jnp.take(one_row, self.safe_variable_id, axis=-1)
            # return one_row[self.variableId]

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """

        @check_shape
        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            model = pm.modelcontext(None)  # active model
            try:
                values = dataframe[self.name].to_numpy()
            except KeyError as e:
                raise BiogemeError(
                    f"Column '{self.name}' not found in the dataframe."
                ) from e

            existing = model.named_vars.get(self.name)
            if existing is not None:
                # Ensure the pm.Data is refreshed with the current values
                pm.set_data({self.name: values}, model=model)
                return existing

            return pm.Data(
                self.name,
                values,
                dims=Dimension.OBS,
            )

        return builder
