"""Representation of unknown parameters

:author: Michel Bierlaire
:date: Sat Apr 20 14:54:16 2024

"""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd
import pymc as pm
from biogeme.exceptions import BiogemeError
from jax import numpy as jnp
from pytensor.tensor import TensorVariable

from .bayesian import PymcModelBuilderType
from .elementary_expressions import Elementary
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)

DEFAULT_SIGMA_PRIOR = 5.0


class PriorFactory(Protocol):
    """
    Protocol for user-defined priors used with :class:`Beta`.

    A prior factory is a callable that receives:

    :param name:
        The PyMC name of the parameter (string).

    :param initial_value:
        The initial value provided by the Biogeme model.

    :param lower_bound:
        The lower bound of the truncation or None if no lower bound

    :param upper_bound:
        The upper bound of the truncation, or None if no upper bound.

    and must return a **PyMC distribution** (RandomVariable) suitable as a prior
    for that parameter. The returned object must be a valid PyMC *distribution
    node*, not a sampled value.

    Example
    -------

    Below is an example of a custom prior that uses a **Student-t distribution**
    and truncates it to enforce **negative support**. This is useful for
    parameters such as cost or travel-time sensitivities, where the coefficient
    is known a priori to be negative.

    .. code-block:: python

        import pymc as pm
        from pytensor.tensor import TensorVariable

        def negative_student_prior(
            name: str,
            initial_value: float,
            lower_bound: float | None,
            upper_bound: float | None,
        ) -> TensorVariable:
            base = pm.StudentT.dist(mu=0.0, sigma=10.0, nu=5.0)
            # Lower bound is ignored.
            # Upper bound is enforced to 0, provided or not.
            upper = 0.0 if upper_bound is None else min(0.0, upper_bound)
            return pm.Truncated(
                name=name,
                dist=base,
                upper=upper,
                initval=initial_value,
            )

    This function can then be passed to :class:`Beta` as:

    .. code-block:: python

        b_cost = Beta(
            'b_cost',
            value=-1.0,
            lowerbound=None,
            upperbound=None,
            status=0,
            prior=negative_student_prior,
        )
    """

    def __call__(
        self, name: str, initial_value: float, lower_bound: float, upper_bound: float
    ) -> TensorVariable: ...


class Beta(Elementary):
    """
    Unknown parameters to be estimated from data.
    """

    def __init__(
        self,
        name: str,
        value: float,
        lowerbound: float | None,
        upperbound: float | None,
        status: int,
        sigma_prior: float = DEFAULT_SIGMA_PRIOR,
        prior: PriorFactory | None = None,
    ):
        """Constructor

        :param name: name of the parameter.
        :param value: default value.
        :param lowerbound: if different from None, imposes a lower
          bound on the value of the parameter during the optimization.
        :param upperbound: if different from None, imposes an upper
          bound on the value of the parameter during the optimization.
        :param status: if different from 0, the parameter is fixed to
          its default value, and not modified by the optimization algorithm.
        :param prior: prior distribution for Bayesian estimation. If None, it will be based on the bounds.

        :raise BiogemeError: if the first parameter is not a str.

        :raise BiogemeError: if the second parameter is not an int or a float.
        """

        if not isinstance(value, (int, float)):
            error_msg = (
                f"The second parameter for {name} must be "
                f"a float and not a {type(value)}: {value}"
            )
            raise BiogemeError(error_msg)
        if not isinstance(name, str):
            error_msg = (
                f"The first parameter must be a string and "
                f"not a {type(name)}: {name}"
            )
            raise BiogemeError(error_msg)
        super().__init__(name)
        self.init_value = value
        self.lower_bound = lowerbound
        self.upper_bound = upperbound
        self.status = status
        self.sigma_prior = sigma_prior
        self.prior = prior

    def deep_flat_copy(self) -> Beta:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(
            name=self.name,
            value=self.init_value,
            lowerbound=self.lower_bound,
            upperbound=self.upper_bound,
            status=self.status,
        )

    @property
    def is_free(self):
        return self.status == 0

    @property
    def is_fixed(self):
        return not self.is_free

    @property
    def expression_type(self) -> TypeOfElementaryExpression:
        """Type of elementary expression"""
        if self.is_free:
            return TypeOfElementaryExpression.FREE_BETA
        return TypeOfElementaryExpression.FIXED_BETA

    def __str__(self) -> str:
        return (
            f"Beta('{self.name}', {self.init_value}, {self.lower_bound}, "
            f"{self.upper_bound}, {self.status})"
        )

    def __repr__(self) -> str:
        return f'<Beta name={self.name} value={self.init_value} status={self.status}>'

    def get_value(self) -> float:
        """Calculates the value of the expression if it is simple"""
        return self.init_value

    @property
    def safe_beta_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for parameter {self.name}")
        return self.specific_id

    def fix_betas(
        self,
        beta_values: dict[str, float],
        prefix: str | None = None,
        suffix: str | None = None,
    ):
        """Fix all the values of the Beta parameters appearing in the
        dictionary

        :param beta_values: dictionary containing the betas to be
            fixed (as key) and their value.
        :type beta_values: dict(str: float)

        :param prefix: if not None, the parameter is renamed, with a
            prefix defined by this argument.
        :type prefix: str

        :param suffix: if not None, the parameter is renamed, with a
            suffix defined by this argument.
        :type suffix: str

        """
        if self.name in beta_values:
            self.init_value = beta_values[self.name]
            self.status = 1
            if prefix is not None:
                self.name = f"{prefix}{self.name}"
            if suffix is not None:
                self.name = f"{self.name}{suffix}"

    def change_init_values(self, betas: dict[str, float]):
        """Modifies the initial values of the Beta parameters.

        The fact that the parameters are fixed or free is irrelevant here.

        :param betas: dictionary where the keys are the names of the
                      parameters, and the values are the new value for
                      the parameters.
        :type betas: dict(string:float)
        """

        value = betas.get(self.name)
        if value is not None and value != self.init_value:
            if self.is_fixed:
                warning_msg = (
                    f'Parameter {self.name} is fixed, but its value '
                    f'is changed from {self.init_value} to {value}.'
                )
                logger.warning(warning_msg)
            self.init_value = value

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Returns a compiled JAX-compatible function that extracts the beta value from the parameter vector using its
        unique index.
        """
        if self.is_free:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.array:
                return jnp.asarray(parameters[self.safe_beta_id])

            return the_jax_function

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.array:
            return jnp.asarray(self.init_value)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Build a PyMC node for a Beta parameter:
        - If free: return the (scalar) RV named `self.name` (create it if missing).
        - If fixed: return a scalar constant tensor (no named var collision).
        """

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            model = pm.modelcontext(None)

            if self.is_fixed:
                # Return a plain scalar constant; do NOT register a pm.Data with the same name.
                return pm.math.constant(float(self.init_value))

            # Free parameter path
            # Check whether a FREE RV with this name already exists
            free_names = {rv.name for rv in model.free_RVs}
            if self.name in free_names:
                # Return the existing scalar RV
                return model.named_vars[self.name]

            # Create the prior according to bounds/prior
            if self.prior is not None:
                rv = self.prior(
                    self.name,
                    initial_value=self.init_value,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                )
            else:
                if self.lower_bound is None and self.upper_bound is None:
                    rv = pm.Normal(
                        self.name, mu=self.init_value, sigma=self.sigma_prior
                    )
                else:
                    rv = pm.TruncatedNormal(
                        self.name,
                        mu=self.init_value,
                        sigma=self.sigma_prior,
                        lower=self.lower_bound,
                        upper=self.upper_bound,
                    )
            return rv

        return builder
