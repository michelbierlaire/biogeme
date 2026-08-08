"""Arithmetic expressions accepted by Biogeme: nested logit."""

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
    from biogeme.nests import NestsForNestedLogit, OldNestsForNestedLogit

    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


def index_of(key: float, keys: jnp.ndarray) -> jnp.ndarray:
    """Return the index of a key in a vector of alternative identifiers."""
    return jnp.argmax(keys == key)


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

        if self.av is not None and self.av[choice].get_value() == 0.0:
            return -np.inf

        mus = np.asarray(
            [nest_parameter.get_value() for nest_parameter in self.nest_parameters],
            dtype=float,
        )

        global_mu = None if self.mu is None else self.mu.get_value()

        kernels = np.full(self.number_of_alternatives, -np.inf, dtype=float)

        for m in range(self.number_of_nests):
            mu_m = mus[m]
            membership_m = self.nest_membership[m, :]

            biosum = np.sum(membership_m * availabilities * np.exp(mu_m * utilities))

            if biosum <= 0.0:
                continue

            log_biosum = np.log(biosum)

            for i in range(self.number_of_alternatives):
                if membership_m[i] == 0.0:
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
        nest_membership = jnp.asarray(self.nest_membership, dtype=JAX_FLOAT)
        alone_membership = jnp.asarray(self.alone_membership, dtype=JAX_FLOAT)

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

            mu_u = mus[:, None] * utilities[None, :]

            masked_terms = jnp.where(
                nest_membership != 0.0,
                mu_u,
                -jnp.inf,
            )

            availability_mask = jnp.where(
                availabilities[None, :] != 0.0,
                0.0,
                -jnp.inf,
            )

            log_biosums = jax.nn.logsumexp(
                masked_terms + availability_mask,
                axis=1,
            )

            if numerically_safe:
                # A nest with no available alternatives has a log biosum of
                # -inf.  Avoid expressions such as 0 * (-inf), and make sure
                # that an unavailable nest cannot contribute to the kernels.
                active_nests = jnp.isfinite(log_biosums)
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
                    (nest_membership != 0.0) & active_nests[:, None],
                    nest_kernels,
                    -jnp.inf,
                )
            else:
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

                nest_kernels = jnp.where(
                    nest_membership != 0.0,
                    nest_kernels,
                    -jnp.inf,
                )

            kernels_from_nests = jax.nn.logsumexp(nest_kernels, axis=0)

            if global_mu is None:
                alone_kernels = utilities
            else:
                alone_kernels = jnp.log(global_mu) + global_mu * utilities

            kernels = jnp.where(
                alone_membership != 0.0,
                alone_kernels,
                kernels_from_nests,
            )

            if numerically_safe:
                positive_availabilities = availabilities > 0.0
                safe_availabilities = jnp.where(
                    positive_availabilities,
                    availabilities,
                    1.0,
                )
                log_availabilities = jnp.where(
                    positive_availabilities,
                    jnp.log(safe_availabilities),
                    -jnp.inf,
                )
                log_denominator = jax.nn.logsumexp(
                    log_availabilities + kernels
                )
                # If every alternative is unavailable, the chosen-availability
                # branch below returns the sentinel value.  Keep the unused
                # arithmetic finite to avoid propagating inf/nan derivatives.
                safe_log_denominator = jnp.where(
                    jnp.isfinite(log_denominator),
                    log_denominator,
                    0.0,
                )
                log_probability = kernels[choice_index] - safe_log_denominator
            else:
                denominator = jnp.sum(availabilities * jnp.exp(kernels))
                log_probability = kernels[choice_index] - jnp.log(denominator)

            unavailable_value = -jnp.finfo(JAX_FLOAT).max

            return jnp.where(
                chosen_availability == 0.0,
                unavailable_value,
                log_probability,
            )

        return the_jax_function
