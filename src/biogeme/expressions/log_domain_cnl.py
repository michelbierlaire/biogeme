"""Numerically safe CNL backend that remains in the log domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from biogeme.floating_point import JAX_FLOAT, LOG_CLIP_MIN, NEGATIVE_LARGE

from .jax_utils import JaxFunctionType
from .log_cross_nested import LogCrossNested
from .numeric_expressions import Numeric


class LogDomainLogCrossNested(LogCrossNested):
    """Internal CNL expression using log-sum-exp throughout its JAX kernel."""

    def recursive_construct_jax_function(
        self,
        numerically_safe: bool,
    ) -> JaxFunctionType:
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
            tuple(
                alpha.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                )
                for alpha in alpha_row
            )
            for alpha_row in self.alpha_matrix
        )
        structural_membership_mask = jnp.asarray(
            [
                [
                    not (isinstance(alpha, Numeric) and alpha.value == 0.0)
                    for alpha in alpha_row
                ]
                for alpha_row in self.alpha_matrix
            ],
            dtype=bool,
        )
        alt_keys = self.alt_keys

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
            alpha_rows = [
                evaluate_all(
                    row_functions,
                    parameters,
                    one_row,
                    the_draws,
                    the_random_variables,
                )
                for row_functions in alpha_functions
            ]
            alphas = jnp.stack(alpha_rows, axis=0)
            global_mu = (
                None
                if mu_function is None
                else mu_function(
                    parameters, one_row, the_draws, the_random_variables
                )
            )

            effective_membership = structural_membership_mask & (alphas > 0.0)
            safe_alphas = jnp.where(
                effective_membership,
                jnp.maximum(alphas, LOG_CLIP_MIN),
                1.0,
            )
            log_alphas = jnp.log(safe_alphas)
            mu_utilities = mus[:, None] * utilities[None, :]
            alpha_exponents = (
                mus[:, None]
                if global_mu is None
                else mus[:, None] / global_mu
            )
            active_memberships = effective_membership & available[None, :]
            active_nests = jnp.any(active_memberships, axis=1)
            log_nest_terms = alpha_exponents * log_alphas + mu_utilities
            log_nest_terms = jnp.where(
                active_memberships,
                log_nest_terms,
                NEGATIVE_LARGE,
            )
            log_biosums = jax.nn.logsumexp(log_nest_terms, axis=1)
            coefficients = (
                (1.0 - mus) / mus
                if global_mu is None
                else (global_mu / mus) - 1.0
            )
            safe_log_biosums = jnp.where(active_nests, log_biosums, 0.0)
            kernel_terms = (
                alpha_exponents * log_alphas
                + mu_utilities
                + coefficients[:, None] * safe_log_biosums[:, None]
            )
            kernel_terms = jnp.where(
                effective_membership & active_nests[:, None],
                kernel_terms,
                NEGATIVE_LARGE,
            )
            kernels = jax.nn.logsumexp(kernel_terms, axis=0)
            if global_mu is not None:
                kernels = jnp.log(global_mu) + kernels

            supported_alternatives = jnp.any(
                effective_membership & active_nests[:, None], axis=0
            )
            denominator_terms = jnp.where(
                available & supported_alternatives,
                kernels,
                NEGATIVE_LARGE,
            )
            log_denominator = jax.nn.logsumexp(denominator_terms)
            log_probability = kernels[choice_index] - log_denominator
            unavailable_value = jnp.asarray(NEGATIVE_LARGE, dtype=JAX_FLOAT)
            valid_choice = (
                any_match
                & chosen_availability
                & supported_alternatives[choice_index]
            )
            return jnp.where(
                valid_choice, log_probability, unavailable_value
            )

        return the_jax_function
