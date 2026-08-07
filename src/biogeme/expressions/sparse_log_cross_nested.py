"""Sparse implementation of the cross-nested logit expression."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from biogeme.floating_point import JAX_FLOAT

from .jax_utils import JaxFunctionType
from .log_cross_nested import LogCrossNested, index_of
from .numeric_expressions import Numeric


class SparseLogCrossNested(LogCrossNested):
    """CNL expression that skips structurally zero allocation parameters."""

    uses_sparse_memberships = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        active_edges: list[tuple[int, int]] = []
        for nest_index, alpha_row in enumerate(self.alpha_matrix):
            for alternative_index, alpha in enumerate(alpha_row):
                if isinstance(alpha, Numeric) and alpha.value == 0.0:
                    continue
                active_edges.append((nest_index, alternative_index))

        self.active_edges = tuple(active_edges)
        self._edge_nest_indices = tuple(edge[0] for edge in self.active_edges)
        self._edge_alternative_indices = tuple(
            edge[1] for edge in self.active_edges
        )

    def recursive_construct_jax_function(
        self,
        numerically_safe: bool,
    ) -> JaxFunctionType:
        """Generate a JAX function operating only on active memberships."""
        if numerically_safe:
            from .log_domain_cnl import LogDomainLogCrossNested

            return LogDomainLogCrossNested.recursive_construct_jax_function(
                self, numerically_safe=True
            )

        utility_functions = tuple(
            utility.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
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
            self.alpha_matrix[nest_index][alternative_index]
            .recursive_construct_jax_function(numerically_safe=numerically_safe)
            for nest_index, alternative_index in self.active_edges
        )

        alt_keys = self.alt_keys
        edge_nest_indices = jnp.asarray(self._edge_nest_indices, dtype=jnp.int32)
        edge_alternative_indices = jnp.asarray(
            self._edge_alternative_indices, dtype=jnp.int32
        )

        def evaluate_all(functions, parameters, row, draws, random_variables):
            return jnp.stack(
                [
                    function(parameters, row, draws, random_variables)
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
            availabilities = (
                jnp.ones_like(utilities)
                if availability_functions is None
                else evaluate_all(
                    availability_functions,
                    parameters,
                    one_row,
                    the_draws,
                    the_random_variables,
                )
            )
            choice_id = choice_function(
                parameters, one_row, the_draws, the_random_variables
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
            alphas = evaluate_all(
                alpha_functions,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )
            global_mu = (
                None
                if mu_function is None
                else mu_function(
                    parameters, one_row, the_draws, the_random_variables
                )
            )

            edge_mus = mus[edge_nest_indices]
            edge_utilities = utilities[edge_alternative_indices]
            edge_availability = availabilities[edge_alternative_indices]
            alpha_exponents = (
                edge_mus if global_mu is None else edge_mus / global_mu
            )
            alpha_power = alphas**alpha_exponents
            contributions = (
                edge_availability
                * alpha_power
                * jnp.exp(edge_mus * edge_utilities)
            )
            biosums = jax.ops.segment_sum(
                contributions,
                edge_nest_indices,
                num_segments=self.number_of_nests,
            )
            log_biosums = jnp.where(
                biosums > 0.0,
                jnp.log(biosums),
                -jnp.inf,
            )
            coefficients = (
                (1.0 - mus) / mus
                if global_mu is None
                else (global_mu / mus) - 1.0
            )
            edge_kernel_terms = (
                edge_mus * edge_utilities
                + coefficients[edge_nest_indices]
                * log_biosums[edge_nest_indices]
            )
            edge_log_kernel = jnp.log(alpha_power) + edge_kernel_terms
            maximum_by_alternative = jax.ops.segment_max(
                edge_log_kernel,
                edge_alternative_indices,
                num_segments=self.number_of_alternatives,
            )
            shifted_exponentials = jnp.exp(
                edge_log_kernel
                - maximum_by_alternative[edge_alternative_indices]
            )
            exponential_sums = jax.ops.segment_sum(
                shifted_exponentials,
                edge_alternative_indices,
                num_segments=self.number_of_alternatives,
            )
            kernels = maximum_by_alternative + jnp.log(exponential_sums)
            if global_mu is not None:
                kernels = jnp.log(global_mu) + kernels

            denominator = jnp.sum(availabilities * jnp.exp(kernels))
            log_probability = kernels[choice_index] - jnp.log(denominator)
            unavailable_value = -jnp.finfo(JAX_FLOAT).max
            return jnp.where(
                chosen_availability == 0.0,
                unavailable_value,
                log_probability,
            )

        return the_jax_function
