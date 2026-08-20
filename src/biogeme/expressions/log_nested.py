"""Arithmetic expressions accepted by Biogeme: nested logit."""

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytensor.tensor as pt

from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT, NEGATIVE_LARGE

from .base_expressions import Expression, LogitTuple
from .bayesian import PymcModelBuilderType
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

if TYPE_CHECKING:
    from biogeme.nests import NestsForNestedLogit, OldNestsForNestedLogit

    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


def index_of(key: float, keys: jnp.ndarray) -> jnp.ndarray:
    """Return the index of a key in a vector of alternative identifiers."""
    return jnp.argmax(keys == key)


def _numpy_logsumexp(values: np.ndarray) -> float:
    """Stable NumPy log-sum-exp for a known non-empty finite vector."""
    return float(np.logaddexp.reduce(np.asarray(values, dtype=float)))


class LogNested(Expression):
    """Log probability of the nested logit model.

    This expression computes the nested logit probability directly, instead of
    expanding it through generic MEV and LogLogit nodes.

    For each nest :math:`m`, define

    .. math::

        B_m = \\sum_{j \\in C_m} a_j \\exp(\\mu_m V_j).

    For an alternative :math:`i \\in C_m`, the standard formulation uses

    .. math::

        H_i =
        \\mu_m V_i
        +
        \\left(\\frac{1}{\\mu_m} - 1\\right) \\log B_m.

    If the optional global homogeneity parameter ``mu`` is provided, the
    explicit-mu formulation uses

    .. math::

        H_i =
        \\log \\mu
        + \\mu_m V_i
        +
        \\left(\\frac{\\mu}{\\mu_m} - 1\\right) \\log B_m.

    The log probability is

    .. math::

        H_y - \\log \\sum_j a_j \\exp(H_j).

    Alternatives that do not belong to any explicit nest are treated as
    singleton alternatives, consistently with ``NestsForNestedLogit.alone``.
    """

    def __init__(
        self,
        util: dict[int, ExpressionOrNumeric],
        av: dict[int, ExpressionOrNumeric] | None,
        nests: NestsForNestedLogit | OldNestsForNestedLogit,
        choice: ExpressionOrNumeric,
        mu: ExpressionOrNumeric | None = None,
    ):
        """Constructor."""
        from biogeme.nests import NestsForNestedLogit

        Expression.__init__(self)

        if not isinstance(nests, NestsForNestedLogit):
            logger.warning(
                'It is recommended to define the nests of the nested logit '
                'model using OneNestForNestedLogit and NestsForNestedLogit.'
            )
            nests = NestsForNestedLogit(
                choice_set=list(util),
                tuple_of_nests=nests,
            )

        ok, message = nests.check_partition()
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

        self.number_of_alternatives = len(self.alt_ids)

        self.nest_names = [nest.name for nest in nests]
        self.number_of_nests = len(self.nest_names)

        self.utility_values = tuple(self.util[i] for i in self.alt_ids)
        self.availability_values = (
            None if self.av is None else tuple(self.av[i] for i in self.alt_ids)
        )

        self.nest_parameters = tuple(
            validate_and_convert(nest.nest_param) for nest in nests
        )

        # nest_membership[m, j] = 1 if alternative j belongs to nest m.
        self.nest_membership = np.asarray(
            [
                [
                    1.0 if alt_id in nest.list_of_alternatives else 0.0
                    for alt_id in self.alt_ids
                ]
                for nest in nests
            ],
            dtype=float,
        )

        # Alternatives outside the explicit nests are handled as singleton nests.
        alone = set() if nests.alone is None else set(nests.alone)
        self.alone_membership = np.asarray(
            [1.0 if alt_id in alone else 0.0 for alt_id in self.alt_ids],
            dtype=float,
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

    def deep_flat_copy(self) -> LogNested:
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

        available = availabilities != 0.0
        if not available[self.alt_ids.index(choice)]:
            return NEGATIVE_LARGE

        mus = np.asarray(
            [nest_parameter.get_value() for nest_parameter in self.nest_parameters],
            dtype=float,
        )

        global_mu = None if self.mu is None else self.mu.get_value()

        kernels = np.full(
            self.number_of_alternatives, NEGATIVE_LARGE, dtype=float
        )

        for m in range(self.number_of_nests):
            mu_m = mus[m]
            membership_m = self.nest_membership[m, :] != 0.0
            active_members = membership_m & available
            if not np.any(active_members):
                continue

            log_biosum = _numpy_logsumexp(
                mu_m * utilities[active_members]
            )

            for i in range(self.number_of_alternatives):
                if not membership_m[i]:
                    continue

                if global_mu is None:
                    kernels[i] = mu_m * utilities[i] + ((1.0 / mu_m) - 1.0) * log_biosum
                else:
                    kernels[i] = (
                        np.log(global_mu)
                        + mu_m * utilities[i]
                        + ((global_mu / mu_m) - 1.0) * log_biosum
                    )

        if np.any(self.alone_membership != 0.0):
            for i in range(self.number_of_alternatives):
                if self.alone_membership[i] == 0.0:
                    continue

                if global_mu is None:
                    kernels[i] = utilities[i]
                else:
                    kernels[i] = np.log(global_mu) + global_mu * utilities[i]

        choice_index = self.alt_ids.index(choice)
        denominator_terms = kernels[available]
        if denominator_terms.size == 0:
            return NEGATIVE_LARGE

        return kernels[choice_index] - _numpy_logsumexp(denominator_terms)

    def __str__(self) -> str:
        util_str = ', '.join(f'{alt}:{expr}' for alt, expr in self.util.items())
        return f'{self.get_class_name()}[choice={self.choice}; U=({util_str})]'

    def recursive_construct_jax_function(
        self,
        numerically_safe: bool,
    ) -> JaxFunctionType:
        """Generate a compact JAX function for nested logit log probability."""

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

        alt_keys = self.alt_keys
        nest_membership = jnp.asarray(self.nest_membership != 0.0, dtype=bool)
        alone_membership = jnp.asarray(self.alone_membership != 0.0, dtype=bool)

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
            choice_matches = choice_id == alt_keys
            any_match = jnp.any(choice_matches)
            choice_index = jnp.argmax(choice_matches)
            available = availabilities != 0.0
            chosen_availability = available[choice_index]

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

            mu_u = mus[:, None] * utilities[None, :]

            if numerically_safe:
                active_memberships = nest_membership & available[None, :]
                active_nests = jnp.any(active_memberships, axis=1)
                masked_terms = jnp.where(
                    active_memberships,
                    mu_u,
                    NEGATIVE_LARGE,
                )
                log_biosums = jax.nn.logsumexp(masked_terms, axis=1)
                safe_log_biosums = jnp.where(active_nests, log_biosums, 0.0)

                if global_mu is None:
                    nest_kernels = (
                        mu_u
                        + ((1.0 / mus) - 1.0)[:, None]
                        * safe_log_biosums[:, None]
                    )
                else:
                    nest_kernels = (
                        jnp.log(global_mu)
                        + mu_u
                        + ((global_mu / mus) - 1.0)[:, None]
                        * safe_log_biosums[:, None]
                    )

                nest_kernels = jnp.where(
                    nest_membership & active_nests[:, None],
                    nest_kernels,
                    NEGATIVE_LARGE,
                )
            else:
                active_memberships = nest_membership & available[None, :]
                biosums = jnp.sum(
                    jnp.exp(
                        jnp.where(
                            active_memberships,
                            mu_u,
                            NEGATIVE_LARGE,
                        )
                    ),
                    axis=1,
                )
                safe_biosums = jnp.where(biosums > 0.0, biosums, 1.0)
                log_biosums = jnp.log(safe_biosums)
                if global_mu is None:
                    nest_kernels = (
                        mu_u + ((1.0 / mus) - 1.0)[:, None] * log_biosums[:, None]
                    )
                else:
                    nest_kernels = (
                        jnp.log(global_mu)
                        + mu_u
                        + ((global_mu / mus) - 1.0)[:, None]
                        * log_biosums[:, None]
                    )

                # A nested-logit alternative belongs to at most one explicit
                # nest, so no log-sum-exp reduction is required here.
                nest_kernels = jnp.where(
                    nest_membership, nest_kernels, 0.0
                )

            kernels_from_nests = (
                jax.nn.logsumexp(nest_kernels, axis=0)
                if numerically_safe
                else jnp.sum(nest_kernels, axis=0)
            )

            if global_mu is None:
                alone_kernels = utilities
            else:
                alone_kernels = jnp.log(global_mu) + global_mu * utilities

            kernels = jnp.where(
                alone_membership,
                alone_kernels,
                kernels_from_nests,
            )

            if numerically_safe:
                denominator_terms = jnp.where(
                    available,
                    kernels,
                    NEGATIVE_LARGE,
                )
                log_denominator = jax.nn.logsumexp(denominator_terms)
                log_probability = kernels[choice_index] - log_denominator
            else:
                denominator_terms = jnp.exp(
                    jnp.where(available, kernels, NEGATIVE_LARGE)
                )
                denominator = jnp.sum(denominator_terms)
                safe_denominator = jnp.where(
                    denominator > 0.0, denominator, 1.0
                )
                log_probability = kernels[choice_index] - jnp.log(
                    safe_denominator
                )

            unavailable_value = jnp.asarray(NEGATIVE_LARGE, dtype=JAX_FLOAT)
            valid_choice = any_match & chosen_availability
            return jnp.where(
                valid_choice, log_probability, unavailable_value
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """Return a vectorized PyTensor builder for nested-logit log probabilities.

        The builder mirrors the compact NumPy and JAX implementations above.  In
        particular, scalar PyMC random variables (nest parameters and the optional
        global homogeneity parameter) are broadcast to all observations, while
        utility, availability, and choice expressions are evaluated row by row.
        """

        utility_builders = tuple(
            utility.recursive_construct_pymc_model_builder()
            for utility in self.utility_values
        )
        availability_builders = (
            None
            if self.availability_values is None
            else tuple(
                availability.recursive_construct_pymc_model_builder()
                for availability in self.availability_values
            )
        )
        choice_builder = self.choice.recursive_construct_pymc_model_builder()
        nest_parameter_builders = tuple(
            nest_parameter.recursive_construct_pymc_model_builder()
            for nest_parameter in self.nest_parameters
        )
        mu_builder = (
            None
            if self.mu is None
            else self.mu.recursive_construct_pymc_model_builder()
        )

        alt_keys = pt.constant(np.asarray(self.alt_ids, dtype=np.int32))
        nest_membership = pt.as_tensor_variable(self.nest_membership)
        alone_membership = pt.as_tensor_variable(self.alone_membership)

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            n_obs = len(dataframe)

            def observation_vector(
                value: pt.TensorVariable, expression_name: str
            ) -> pt.TensorVariable:
                if value.ndim == 0:
                    return pt.ones((n_obs,), dtype=value.dtype) * value
                if value.ndim != 1:
                    raise BiogemeError(
                        'LogNested PyMC builder: '
                        f'{expression_name} must return a scalar or a 1-D tensor; '
                        f'got ndim={value.ndim}'
                    )
                return value

            utilities = pt.stack(
                [
                    observation_vector(utility_builder(dataframe), 'utility')
                    for utility_builder in utility_builders
                ],
                axis=1,
            )

            if availability_builders is None:
                availabilities = pt.ones_like(utilities)
            else:
                availabilities = pt.stack(
                    [
                        observation_vector(
                            availability_builder(dataframe), 'availability'
                        )
                        for availability_builder in availability_builders
                    ],
                    axis=1,
                )
                availabilities = pt.where(
                    ~(pt.isnan(availabilities) | pt.isinf(availabilities)),
                    availabilities,
                    0.0,
                )

            utilities = pt.where(
                ~(pt.isnan(utilities) | pt.isinf(utilities)),
                utilities,
                NEGATIVE_LARGE,
            )

            choice = observation_vector(choice_builder(dataframe), 'choice')
            choice_i32 = pt.cast(choice, 'int32')

            nest_parameters = pt.stack(
                [
                    observation_vector(
                        nest_parameter_builder(dataframe), 'nest parameter'
                    )
                    for nest_parameter_builder in nest_parameter_builders
                ],
                axis=1,
            )
            if mu_builder is None:
                global_mu = None
            else:
                global_mu = observation_vector(mu_builder(dataframe), 'global mu')

            membership = pt.cast(nest_membership, utilities.dtype)
            alone = pt.cast(alone_membership, utilities.dtype)
            negative_large = pt.cast(
                pt.as_tensor_variable(NEGATIVE_LARGE), utilities.dtype
            )

            available = pt.neq(availabilities, 0.0)
            mu_times_utility = nest_parameters[:, :, None] * utilities[:, None, :]
            active_memberships = (
                pt.neq(membership[None, :, :], 0.0) & available[:, None, :]
            )
            masked_mu_times_utility = pt.where(
                active_memberships,
                mu_times_utility,
                negative_large,
            )
            log_biosums = pt.logsumexp(masked_mu_times_utility, axis=2)
            active_nests = pt.any(active_memberships, axis=2)
            safe_log_biosums = pt.where(active_nests, log_biosums, 0.0)

            if global_mu is None:
                nest_kernels = (
                    mu_times_utility
                    + ((1.0 / nest_parameters) - 1.0)[:, :, None]
                    * safe_log_biosums[:, :, None]
                )
            else:
                nest_kernels = (
                    pt.log(global_mu)[:, None, None]
                    + mu_times_utility
                    + ((global_mu[:, None] / nest_parameters) - 1.0)[:, :, None]
                    * safe_log_biosums[:, :, None]
                )

            nest_kernels = pt.where(
                pt.neq(membership[None, :, :], 0.0) & active_nests[:, :, None],
                nest_kernels,
                negative_large,
            )
            kernels_from_nests = pt.logsumexp(nest_kernels, axis=1)

            if global_mu is None:
                alone_kernels = utilities
            else:
                alone_kernels = (
                    pt.log(global_mu)[:, None] + global_mu[:, None] * utilities
                )

            kernels = pt.where(
                pt.neq(alone[None, :], 0.0),
                alone_kernels,
                kernels_from_nests,
            )

            denominator_terms = pt.where(
                available, kernels, negative_large
            )
            log_denominator = pt.logsumexp(denominator_terms, axis=1)

            matches = pt.eq(choice_i32[:, None], alt_keys[None, :])
            choice_index = pt.argmax(matches, axis=1)
            any_match = pt.any(matches, axis=1)
            row_index = pt.arange(n_obs)
            safe_choice_index = pt.where(any_match, choice_index, 0)
            chosen_kernel = kernels[row_index, safe_choice_index]
            chosen_availability = availabilities[row_index, safe_choice_index]
            log_probability = chosen_kernel - log_denominator

            valid_choice = any_match & pt.neq(chosen_availability, 0.0)
            return pt.where(valid_choice, log_probability, negative_large)

        return builder
