"""Sparse implementation of the cross-nested logit expression."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from biogeme.floating_point import JAX_FLOAT, LOG_CLIP_MIN, NEGATIVE_LARGE

from .jax_utils import JaxFunctionType
from .log_cross_nested import LogCrossNested
from .numeric_expressions import Numeric


def _segment_logsumexp(
    values: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    """Compute log-sum-exp by segment without materializing a dense matrix."""
    finite_values = jnp.isfinite(values)
    positive_infinite_values = jnp.isposinf(values)
    finite_counts = jax.ops.segment_sum(
        finite_values.astype(JAX_FLOAT),
        segment_ids,
        num_segments=num_segments,
    )
    positive_infinite_counts = jax.ops.segment_sum(
        positive_infinite_values.astype(JAX_FLOAT),
        segment_ids,
        num_segments=num_segments,
    )

    finite_values_for_max = jnp.where(finite_values, values, -jnp.inf)
    maximum = jax.ops.segment_max(
        finite_values_for_max,
        segment_ids,
        num_segments=num_segments,
    )
    has_finite_values = finite_counts > 0.0
    safe_maximum = jnp.where(has_finite_values, maximum, 0.0)

    shifted = jnp.where(
        finite_values,
        jnp.exp(values - safe_maximum[segment_ids]),
        0.0,
    )
    sums = jax.ops.segment_sum(
        shifted,
        segment_ids,
        num_segments=num_segments,
    )
    finite_result = jnp.where(
        has_finite_values,
        safe_maximum + jnp.log(sums),
        -jnp.inf,
    )
    return jnp.where(
        positive_infinite_counts > 0.0,
        jnp.inf,
        finite_result,
    )


def _safe_segment_logsumexp(
    values: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    """Finite log-sum-exp by segment for differentiated safe expressions.

    A finite dummy term is appended to every segment.  Empty segments then
    evaluate to ``NEGATIVE_LARGE`` without passing infinities or zero sums to
    automatic differentiation.
    """
    dummy_ids = jnp.arange(num_segments, dtype=jnp.int32)
    augmented_ids = jnp.concatenate((segment_ids, dummy_ids))
    augmented_values = jnp.concatenate(
        (
            values,
            jnp.full((num_segments,), NEGATIVE_LARGE, dtype=JAX_FLOAT),
        )
    )
    maximum = jax.ops.segment_max(
        augmented_values,
        augmented_ids,
        num_segments=num_segments,
    )
    shifted = jnp.exp(augmented_values - maximum[augmented_ids])
    sums = jax.ops.segment_sum(
        shifted,
        augmented_ids,
        num_segments=num_segments,
    )
    return maximum + jnp.log(sums)


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

        if numerically_safe:

            def the_safe_jax_function(
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
                alpha_exponents = (
                    edge_mus if global_mu is None else edge_mus / global_mu
                )
                effective_edges = alphas > 0.0
                active_edges = (
                    effective_edges & available[edge_alternative_indices]
                )
                safe_alphas = jnp.where(
                    effective_edges,
                    jnp.maximum(alphas, LOG_CLIP_MIN),
                    1.0,
                )
                log_alphas = jnp.log(safe_alphas)

                edge_log_biosum_terms = (
                    alpha_exponents * log_alphas
                    + edge_mus * edge_utilities
                )
                edge_log_biosum_terms = jnp.where(
                    active_edges,
                    edge_log_biosum_terms,
                    NEGATIVE_LARGE,
                )
                log_biosums = _safe_segment_logsumexp(
                    edge_log_biosum_terms,
                    edge_nest_indices,
                    num_segments=self.number_of_nests,
                )
                active_nests = (
                    jax.ops.segment_sum(
                        active_edges.astype(JAX_FLOAT),
                        edge_nest_indices,
                        num_segments=self.number_of_nests,
                    )
                    > 0.0
                )
                safe_log_biosums = jnp.where(active_nests, log_biosums, 0.0)
                coefficients = (
                    (1.0 - mus) / mus
                    if global_mu is None
                    else (global_mu / mus) - 1.0
                )
                edge_log_kernel = (
                    alpha_exponents * log_alphas
                    + edge_mus * edge_utilities
                    + coefficients[edge_nest_indices]
                    * safe_log_biosums[edge_nest_indices]
                )
                kernel_edges = (
                    effective_edges & active_nests[edge_nest_indices]
                )
                edge_log_kernel = jnp.where(
                    kernel_edges,
                    edge_log_kernel,
                    NEGATIVE_LARGE,
                )
                kernels = _safe_segment_logsumexp(
                    edge_log_kernel,
                    edge_alternative_indices,
                    num_segments=self.number_of_alternatives,
                )
                if global_mu is not None:
                    kernels = jnp.log(global_mu) + kernels

                supported_alternatives = (
                    jax.ops.segment_sum(
                        kernel_edges.astype(JAX_FLOAT),
                        edge_alternative_indices,
                        num_segments=self.number_of_alternatives,
                    )
                    > 0.0
                )
                denominator_terms = jnp.where(
                    available & supported_alternatives,
                    kernels,
                    NEGATIVE_LARGE,
                )
                log_denominator = jax.nn.logsumexp(denominator_terms)
                log_probability = kernels[choice_index] - log_denominator
                unavailable_value = jnp.asarray(
                    NEGATIVE_LARGE, dtype=JAX_FLOAT
                )
                valid_choice = (
                    any_match
                    & chosen_availability
                    & supported_alternatives[choice_index]
                )
                return jnp.where(
                    valid_choice, log_probability, unavailable_value
                )

            return the_safe_jax_function

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
            edge_availability = available[edge_alternative_indices]
            alpha_exponents = (
                edge_mus if global_mu is None else edge_mus / global_mu
            )
            alpha_power = alphas**alpha_exponents
            exponential_arguments = jnp.where(
                edge_availability,
                edge_mus * edge_utilities,
                NEGATIVE_LARGE,
            )
            contributions = alpha_power * jnp.exp(exponential_arguments)
            biosums = jax.ops.segment_sum(
                contributions,
                edge_nest_indices,
                num_segments=self.number_of_nests,
            )
            safe_biosums = jnp.where(biosums > 0.0, biosums, 1.0)
            log_biosums = jnp.log(safe_biosums)
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
            safe_exponential_sums = jnp.where(
                exponential_sums > 0.0, exponential_sums, 1.0
            )
            kernels = maximum_by_alternative + jnp.log(
                safe_exponential_sums
            )
            if global_mu is not None:
                kernels = jnp.log(global_mu) + kernels

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
