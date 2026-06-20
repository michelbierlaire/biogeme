"""Arithmetic expressions accepted by Biogeme: sampled logit."""

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT

from .base_expressions import Expression, LogitTuple
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

if TYPE_CHECKING:
    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


class LogSampledLogit(Expression):
    """Log probability for a sampled multinomial logit model.

    This expression is designed for sampling of alternatives. It represents

    .. math::

        V_0 - \\omega_0
        -
        \\log \\sum_{j \\in S}
        \\exp(V_j - \\omega_j),

    where alternative ``0`` is the chosen alternative by construction, and
    :math:`\\omega_j` is the log of the sampling probability correction stored
    in the generated database.

    This replaces a large expanded expression of the form

    .. code-block:: python

        loglogit(
            {
                i: utility_i - log_sampling_probability_i
                for i in sample
            },
            None,
            0,
        )

    by a single expression node with a compact JAX implementation.
    """

    def __init__(
        self,
        utilities: dict[int, ExpressionOrNumeric],
        log_probabilities: dict[int, ExpressionOrNumeric],
        choice: ExpressionOrNumeric = 0,
    ):
        """Constructor.

        :param utilities: dictionary of utility expressions for sampled
            alternatives, indexed by their position in the generated sample.
            The chosen alternative is expected to be index 0.
        :param log_probabilities: dictionary of log sampling probability
            corrections, indexed by the same sample positions.
        :param choice: expression identifying the chosen alternative in the
            sampled set. Defaults to 0.
        """
        Expression.__init__(self)

        if set(utilities) != set(log_probabilities):
            missing_log_probabilities = set(utilities) - set(log_probabilities)
            unknown_log_probabilities = set(log_probabilities) - set(utilities)
            raise BiogemeError(
                'The utility and log-probability dictionaries must contain '
                'exactly the same sample identifiers. '
                f'Missing log-probability entries: {missing_log_probabilities}. '
                f'Unknown log-probability entries: {unknown_log_probabilities}.'
            )

        if not utilities:
            raise BiogemeError('The dictionary of sampled utilities cannot be empty.')

        self.utilities: dict[int, Expression] = {
            sample_id: validate_and_convert(expression)
            for sample_id, expression in utilities.items()
        }

        self.log_probabilities: dict[int, Expression] = {
            sample_id: validate_and_convert(expression)
            for sample_id, expression in log_probabilities.items()
        }

        self.choice: Expression = validate_and_convert(choice)

        self.sample_ids = list(self.utilities.keys())
        self.sample_keys = jnp.array(self.sample_ids, dtype=JAX_FLOAT)
        self.number_of_sampled_alternatives = len(self.sample_ids)

        self.utility_values = tuple(self.utilities[i] for i in self.sample_ids)
        self.log_probability_values = tuple(
            self.log_probabilities[i] for i in self.sample_ids
        )

        self.children.append(self.choice)

        for expression in self.utility_values:
            self.children.append(expression)

        for expression in self.log_probability_values:
            self.children.append(expression)

    def deep_flat_copy(self) -> LogSampledLogit:
        """Deep flat copy."""
        copy_utilities = {
            sample_id: utility.deep_flat_copy()
            for sample_id, utility in self.utilities.items()
        }
        copy_log_probabilities = {
            sample_id: log_probability.deep_flat_copy()
            for sample_id, log_probability in self.log_probabilities.items()
        }
        copy_choice = self.choice.deep_flat_copy()

        return type(self)(
            utilities=copy_utilities,
            log_probabilities=copy_log_probabilities,
            choice=copy_choice,
        )

    def logit_choice_avail(self) -> list[LogitTuple]:
        """Return availability structures appearing in this expression."""
        return list(
            chain.from_iterable(child.logit_choice_avail() for child in self.children)
        )

    def get_value(self) -> float:
        """Evaluate the sampled logit log probability using NumPy."""
        choice = int(self.choice.get_value())

        if choice not in self.utilities:
            raise BiogemeError(
                f'Alternative {choice} does not appear in the sampled utilities: '
                f'{self.utilities.keys()}'
            )

        kernels = np.asarray(
            [
                self.utilities[sample_id].get_value()
                - self.log_probabilities[sample_id].get_value()
                for sample_id in self.sample_ids
            ],
            dtype=float,
        )

        choice_index = self.sample_ids.index(choice)
        chosen_kernel = kernels[choice_index]

        denominator = np.log(np.sum(np.exp(kernels - chosen_kernel)))

        return -denominator

    def __str__(self) -> str:
        entries = ', '.join(
            f'{sample_id}:{self.utilities[sample_id]}-{self.log_probabilities[sample_id]}'
            for sample_id in self.sample_ids
        )
        return f'{self.get_class_name()}[choice={self.choice}; kernels=({entries})]'

    def recursive_construct_jax_function(
        self,
        numerically_safe: bool,
    ) -> JaxFunctionType:
        """Generate a compact JAX function for sampled logit."""

        utility_functions = tuple(
            utility.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for utility in self.utility_values
        )

        log_probability_functions = tuple(
            log_probability.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
            for log_probability in self.log_probability_values
        )

        choice_function = self.choice.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        sample_keys = self.sample_keys

        def evaluate_all(
            functions,
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.stack(
                [
                    function(parameters, one_row, the_draws, the_random_variables)
                    for function in functions
                ],
                axis=0,
            )

        def index_of(key: jnp.ndarray) -> jnp.ndarray:
            return jnp.argmax(sample_keys == key)

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            utilities = evaluate_all(
                utility_functions,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )

            log_probabilities = evaluate_all(
                log_probability_functions,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )

            kernels = utilities - log_probabilities

            choice_id = choice_function(
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )
            choice_index = index_of(choice_id)

            chosen_kernel = kernels[choice_index]

            return chosen_kernel - jax.nn.logsumexp(kernels)

        return the_jax_function
