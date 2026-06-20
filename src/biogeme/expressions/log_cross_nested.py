"""Arithmetic expressions accepted by Biogeme: cross-nested logit."""

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
    from biogeme.nests import NestsForCrossNestedLogit, OldNestsForCrossNestedLogit

    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


def index_of(key: float, keys: jnp.ndarray) -> jnp.ndarray:
    """Return the index of a key in a vector of alternative identifiers."""
    return jnp.argmax(keys == key)


class LogCrossNested(Expression):
    """Log probability of the cross-nested logit model.

    This expression computes the CNL probability directly, instead of
    expanding it through generic WeightedLogSumExp and LogLogit nodes.

    It implements:

    .. math::

        H_i =
        \\log \\sum_m
        \\alpha_{im}^{\\mu_m}
        \\exp(\\mu_m V_i)
        B_m^{(1-\\mu_m)/\\mu_m}

    where

    .. math::

        B_m =
        \\sum_j a_j \\alpha_{jm}^{\\mu_m} \\exp(\\mu_m V_j).

    The log probability is then

    .. math::

        H_y - \\log \\sum_j a_j \\exp(H_j).

    If the optional global homogeneity parameter ``mu`` is provided, the
    explicit-mu CNL formulation is used. If ``mu`` is ``None``, the standard
    formulation is used.
    """

    def __init__(
        self,
        util: dict[int, ExpressionOrNumeric],
        av: dict[int, ExpressionOrNumeric] | None,
        nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
        choice: ExpressionOrNumeric,
        mu: ExpressionOrNumeric | None = None,
    ):
        """Constructor."""
        from biogeme.nests import NestsForCrossNestedLogit

        Expression.__init__(self)

        if not isinstance(nests, NestsForCrossNestedLogit):
            logger.warning(
                'It is recommended to define the nests of the cross-nested '
                'logit model using OneNestForNestedLogit and '
                'NestsForCrossNestedLogit.'
            )
            nests = NestsForCrossNestedLogit(
                choice_set=list(util),
                tuple_of_nests=nests,
            )

        ok, message = nests.check_validity()
        if not ok:
            raise BiogemeError(message)

        self._is_complex = True
        self.util: dict[int, Expression] = {
            alt_id: validate_and_convert(expression)
            for alt_id, expression in util.items()
        }

        self.av: dict[int, Expression] | None = None
        if av is not None:
            self.av = {
                alt_id: validate_and_convert(expression)
                for alt_id, expression in av.items()
            }

            missing_availability = set(self.util) - set(self.av)
            unknown_availability = set(self.av) - set(self.util)
            if missing_availability or unknown_availability:
                raise BiogemeError(
                    'The availability dictionary must contain exactly the same '
                    'alternative identifiers as the utility dictionary. '
                    f'Missing entries: {missing_availability}. '
                    f'Unknown entries: {unknown_availability}.'
                )

        self.nests = nests
        self.choice: Expression = validate_and_convert(choice)
        self.mu: Expression | None = None if mu is None else validate_and_convert(mu)

        self.alt_ids = list(self.util.keys())
        self.alt_keys = jnp.array(self.alt_ids, dtype=JAX_FLOAT)

        self.nest_names = [nest.name for nest in nests]
        self.number_of_alternatives = len(self.alt_ids)
        self.number_of_nests = len(self.nest_names)

        self.utility_values = tuple(self.util[i] for i in self.alt_ids)
        self.availability_values = (
            None if self.av is None else tuple(self.av[i] for i in self.alt_ids)
        )

        self.nest_parameters = tuple(
            validate_and_convert(nest.nest_param) for nest in nests
        )

        self.alpha_matrix: tuple[tuple[Expression, ...], ...] = tuple(
            tuple(
                validate_and_convert(nest.dict_of_alpha.get(alt_id, 0.0))
                for alt_id in self.alt_ids
            )
            for nest in nests
        )

        if self.av is not None:
            for expression in self.availability_values:
                self.children.append(expression)

        self.children.append(self.choice)
        if self.mu is not None:
            self.children.append(self.mu)

        for expression in self.utility_values:
            self.children.append(expression)

        for expression in self.nest_parameters:
            self.children.append(expression)

        for nest_alpha_values in self.alpha_matrix:
            for expression in nest_alpha_values:
                self.children.append(expression)

    def deep_flat_copy(self) -> LogCrossNested:
        """Deep flat copy."""
        copy_util = {
            alt_id: utility.deep_flat_copy() for alt_id, utility in self.util.items()
        }

        copy_av = (
            {
                alt_id: availability.deep_flat_copy()
                for alt_id, availability in self.av.items()
            }
            if self.av is not None
            else None
        )

        copy_choice = self.choice.deep_flat_copy()
        copy_mu = None if self.mu is None else self.mu.deep_flat_copy()

        # Reconstructing nests generically is intentionally avoided here.
        # For the first implementation, the expression is copied with the
        # same nest object. The nest expressions themselves are already
        # part of the expression children.
        return type(self)(
            util=copy_util,
            av=copy_av,
            nests=self.nests,
            choice=copy_choice,
            mu=copy_mu,
        )

    def logit_choice_avail(self) -> list[LogitTuple]:
        """Return availability structures appearing in this expression."""
        result: list[LogitTuple] = list(
            chain.from_iterable(child.logit_choice_avail() for child in self.children)
        )
        if self.av is not None:
            result.append(LogitTuple(choice=self.choice, availabilities=self.av))
        return result

    def get_value(self) -> float:
        """Evaluate the expression using NumPy."""
        choice = int(self.choice.get_value())

        if choice not in self.util:
            raise BiogemeError(
                f'Alternative {choice} does not appear in the utilities: '
                f'{self.util.keys()}'
            )

        utilities = np.asarray(
            [utility.get_value() for utility in self.utility_values],
            dtype=float,
        )

        if self.av is None:
            availabilities = np.ones(self.number_of_alternatives, dtype=float)
        else:
            availabilities = np.asarray(
                [availability.get_value() for availability in self.availability_values],
                dtype=float,
            )

        if self.av is not None and self.av[choice].get_value() == 0.0:
            return -np.inf

        mus = np.asarray(
            [nest_parameter.get_value() for nest_parameter in self.nest_parameters],
            dtype=float,
        )

        alphas = np.asarray(
            [
                [alpha.get_value() for alpha in alpha_row]
                for alpha_row in self.alpha_matrix
            ],
            dtype=float,
        )

        global_mu = None if self.mu is None else self.mu.get_value()

        if global_mu is None:
            kernels = np.full(self.number_of_alternatives, -np.inf, dtype=float)

            for m in range(self.number_of_nests):
                mu_m = mus[m]
                alpha_m = alphas[m, :]

                biosum = np.sum(
                    availabilities * alpha_m**mu_m * np.exp(mu_m * utilities)
                )

                if biosum <= 0.0:
                    continue

                log_biosum = np.log(biosum)

                for i in range(self.number_of_alternatives):
                    if alpha_m[i] == 0.0:
                        continue

                    term = (
                        mu_m * np.log(alpha_m[i])
                        + mu_m * utilities[i]
                        + ((1.0 - mu_m) / mu_m) * log_biosum
                    )

                    kernels[i] = np.logaddexp(kernels[i], term)
        else:
            kernels = np.full(self.number_of_alternatives, -np.inf, dtype=float)

            for m in range(self.number_of_nests):
                mu_m = mus[m]
                alpha_m = alphas[m, :]
                alpha_exponent = mu_m / global_mu

                biosum = np.sum(
                    availabilities * alpha_m**alpha_exponent * np.exp(mu_m * utilities)
                )

                if biosum <= 0.0:
                    continue

                log_biosum = np.log(biosum)

                for i in range(self.number_of_alternatives):
                    if alpha_m[i] == 0.0:
                        continue

                    term = (
                        alpha_exponent * np.log(alpha_m[i])
                        + mu_m * utilities[i]
                        + ((global_mu / mu_m) - 1.0) * log_biosum
                    )

                    kernels[i] = np.logaddexp(kernels[i], term)

            kernels = np.log(global_mu) + kernels

        choice_index = self.alt_ids.index(choice)
        denominator = np.sum(availabilities * np.exp(kernels))

        if denominator <= 0.0:
            return -np.inf

        return kernels[choice_index] - np.log(denominator)

    def __str__(self) -> str:
        util_str = ', '.join(f'{alt}:{expr}' for alt, expr in self.util.items())
        return f'{self.get_class_name()}[choice={self.choice}; U=({util_str})]'

    def recursive_construct_jax_function(
        self,
        numerically_safe: bool,
    ) -> JaxFunctionType:
        """Generate a compact JAX function for CNL log probability."""

        utility_functions = tuple(
            utility.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for utility in self.utility_values
        )

        availability_functions = (
            None
            if self.availability_values is None
            else tuple(
                availability.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                )
                for availability in self.availability_values
            )
        )

        choice_function = self.choice.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        mu_function = (
            None
            if self.mu is None
            else self.mu.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
        )

        nest_parameter_functions = tuple(
            nest_parameter.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
            for nest_parameter in self.nest_parameters
        )

        alpha_functions = tuple(
            tuple(
                alpha.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                )
                for alpha in alpha_row
            )
            for alpha_row in self.alpha_matrix
        )

        alt_keys = self.alt_keys

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

            if availability_functions is None:
                availabilities = jnp.ones_like(utilities)
            else:
                availabilities = evaluate_all(
                    availability_functions,
                    parameters,
                    one_row,
                    the_draws,
                    the_random_variables,
                )

            choice_id = choice_function(
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )
            choice_index = index_of(choice_id, alt_keys)
            chosen_availability = availabilities[choice_index]

            mus = evaluate_all(
                nest_parameter_functions,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )

            global_mu = (
                None
                if mu_function is None
                else mu_function(
                    parameters,
                    one_row,
                    the_draws,
                    the_random_variables,
                )
            )

            alpha_rows = [
                evaluate_all(
                    alpha_row_functions,
                    parameters,
                    one_row,
                    the_draws,
                    the_random_variables,
                )
                for alpha_row_functions in alpha_functions
            ]
            alphas = jnp.stack(alpha_rows, axis=0)

            # Shape conventions:
            # utilities:      (J,)
            # availabilities: (J,)
            # mus:            (M,)
            # alphas:         (M, J)

            mu_u = mus[:, None] * utilities[None, :]

            if global_mu is None:
                alpha_power = alphas ** mus[:, None]

                biosums = jnp.sum(
                    availabilities[None, :] * alpha_power * jnp.exp(mu_u),
                    axis=1,
                )

                log_biosums = jnp.where(
                    biosums > 0.0,
                    jnp.log(biosums),
                    -jnp.inf,
                )

                kernel_terms = (
                    mu_u + ((1.0 - mus) / mus)[:, None] * log_biosums[:, None]
                )

                kernel_weights = alpha_power

                kernels = jax.nn.logsumexp(
                    kernel_terms,
                    axis=0,
                    b=kernel_weights,
                )
            else:
                alpha_exponents = mus / global_mu
                alpha_power = alphas ** alpha_exponents[:, None]

                biosums = jnp.sum(
                    availabilities[None, :] * alpha_power * jnp.exp(mu_u),
                    axis=1,
                )

                log_biosums = jnp.where(
                    biosums > 0.0,
                    jnp.log(biosums),
                    -jnp.inf,
                )

                kernel_terms = (
                    mu_u + ((global_mu / mus) - 1.0)[:, None] * log_biosums[:, None]
                )

                kernel_weights = alpha_power

                kernels = jnp.log(global_mu) + jax.nn.logsumexp(
                    kernel_terms,
                    axis=0,
                    b=kernel_weights,
                )

            denominator = jnp.sum(availabilities * jnp.exp(kernels))

            log_probability = kernels[choice_index] - jnp.log(denominator)

            unavailable_value = -jnp.finfo(JAX_FLOAT).max

            return jnp.where(
                chosen_availability == 0.0,
                unavailable_value,
                log_probability,
            )

        return the_jax_function
