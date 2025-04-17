"""
Module in charge of the actual calculation of the formula on the database.

Michel Bierlaire
Wed Mar 26 19:30:57 2025
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import build_vectorized_function, Expression
from biogeme.floating_point import JAX_FLOAT, NUMPY_FLOAT
from biogeme.function_output import FunctionOutput
from biogeme.model_elements import ModelElements


class CompiledFormulaEvaluator:
    """
    Compiles and evaluates a Biogeme expression using JAX for efficient repeated computation.
    """

    def __init__(self, model_elements: ModelElements):
        """
        Prepares and compiles the JAX function for evaluating a Biogeme expression.

        :param model_elements: All elements needed to calculate the expression.
        """
        from biogeme.expressions import build_vectorized_function

        self.model_elements = model_elements
        self.free_betas_names = (
            self.model_elements.expressions_registry.free_betas_names
        )
        self.data_jax = self.model_elements.database.data_jax
        self.draws_jax = self.model_elements.draws_management.draws_jax
        n_obs = self.model_elements.sample_size
        n_rv = self.model_elements.expressions_registry.number_of_random_variables
        self.random_variables_jax = jnp.zeros((n_rv,), dtype=JAX_FLOAT)

        log_likelihood = self.model_elements.loglikelihood
        if log_likelihood is None:
            error_message = f'No expression found for log likelihood. Available expressions: {self.model_elements.formula_names}'
            raise BiogemeError(error_message)
        the_function = log_likelihood.recursive_construct_jax_function()
        vectorized_function = build_vectorized_function(the_function)

        if self.model_elements.weight is not None:
            weight_function = (
                self.model_elements.weight.recursive_construct_jax_function()
            )
            vectorized_weight_function = build_vectorized_function(weight_function)
        else:
            vectorized_weight_function = None

        @jit
        def sum_function(
            params: list[float],
            data: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            values = vectorized_function(params, data, draws, random_variables)
            if vectorized_weight_function is not None:
                weights = vectorized_weight_function(
                    params, data, draws, random_variables
                )
                values *= weights
            return jnp.asarray(jnp.sum(values), dtype=JAX_FLOAT), values

        self.sum_function = sum_function

    def evaluate(
        self,
        the_betas: dict[str, float],
        gradient: bool,
        hessian: bool,
        bhhh: bool,
    ) -> FunctionOutput:
        free_betas_values = (
            self.model_elements.expressions_registry.get_complete_betas_array(
                betas_dict=the_betas
            )
        )

        if not gradient:
            return self._evaluate_function_only(free_betas_values)

        if bhhh:
            if hessian:
                return self._evaluate_autodiff_hessian_bhhh(free_betas_values)
            else:
                return self._evaluate_bhhh_only(free_betas_values)

        if hessian:
            return self._evaluate_autodiff_hessian(free_betas_values)

        return self._evaluate_function_and_gradient(free_betas_values)

    def _evaluate_function_only(self, free_betas_values):
        value_jax, _ = self.sum_function(
            free_betas_values,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )
        return FunctionOutput(
            function=float(value_jax),
            gradient=None,
            hessian=None,
            bhhh=None,
        )

    def _evaluate_function_and_gradient(self, free_betas_values):
        value_and_grad_fn = jax.jit(
            jax.value_and_grad(
                lambda p, d, r, rv: self.sum_function(p, d, r, rv)[0], argnums=0
            )
        )
        value, the_gradient = value_and_grad_fn(
            free_betas_values,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )
        return FunctionOutput(
            function=float(value),
            gradient=np.asarray(the_gradient, dtype=NUMPY_FLOAT),
            hessian=None,
            bhhh=None,
        )

    def _evaluate_autodiff_hessian(self, free_betas_values):
        value_and_grad_fn = jax.jit(
            jax.value_and_grad(
                lambda p, d, r, rv: self.sum_function(p, d, r, rv)[0], argnums=0
            )
        )
        value, the_gradient = value_and_grad_fn(
            free_betas_values,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )

        if jnp.all(the_gradient == 0.0):
            the_hessian = jnp.zeros(
                (len(free_betas_values), len(free_betas_values)), dtype=JAX_FLOAT
            )
        else:
            hessian_fn = jax.jit(
                jax.jacfwd(
                    jax.grad(
                        lambda p, d, r, rv: self.sum_function(p, d, r, rv)[0], argnums=0
                    ),
                    argnums=0,
                )
            )
            hess_autodiff = hessian_fn(
                free_betas_values,
                self.data_jax,
                self.draws_jax,
                self.random_variables_jax,
            )
            if jnp.any(jnp.isnan(hess_autodiff)):
                the_hessian = self._evaluate_finite_difference_hessian(
                    free_betas_values
                )
            else:
                the_hessian = np.asarray(hess_autodiff, dtype=NUMPY_FLOAT)

        return FunctionOutput(
            function=float(value),
            gradient=np.asarray(the_gradient, dtype=NUMPY_FLOAT),
            hessian=the_hessian,
            bhhh=None,
        )

    def _evaluate_bhhh_only(self, free_betas_values):
        _, individual_values = self.sum_function(
            free_betas_values,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )

        def one_gradient(p, d, r, rv):
            loglik_fn = (
                self.model_elements.loglikelihood.recursive_construct_jax_function()
            )
            vectorized_function = build_vectorized_function(loglik_fn)
            draw_values = vectorized_function(p, d[None, :], r[None, :, :], rv)
            return jnp.mean(draw_values)

        per_obs_grad_fn = jax.jit(
            jax.vmap(jax.grad(one_gradient, argnums=0), in_axes=(None, 0, 0, 0))
        )
        free_betas_values_jnp = jnp.asarray(free_betas_values, dtype=JAX_FLOAT)
        random_variables_broadcast = jnp.tile(
            self.random_variables_jax[None, :], (self.data_jax.shape[0], 1)
        )
        individual_gradients = per_obs_grad_fn(
            free_betas_values_jnp,
            self.data_jax,
            self.draws_jax,
            random_variables_broadcast,
        )

        expected_shape = (self.data_jax.shape[0], len(free_betas_values))
        if individual_gradients.shape != expected_shape:
            individual_gradients = jnp.tile(
                individual_gradients, (self.data_jax.shape[0], 1)
            )

        the_gradient = jnp.sum(individual_gradients, axis=0)
        bhhh_matrix = jnp.sum(
            jnp.stack([jnp.outer(g, g) for g in individual_gradients]), axis=0
        )

        return FunctionOutput(
            function=float(jnp.sum(individual_values)),
            gradient=np.asarray(the_gradient, dtype=NUMPY_FLOAT),
            hessian=None,
            bhhh=np.asarray(bhhh_matrix, dtype=NUMPY_FLOAT),
        )

    def _evaluate_autodiff_hessian_bhhh(self, free_betas_values):
        bhhh_result = self._evaluate_bhhh_only(free_betas_values)
        hessian = self._evaluate_autodiff_hessian(free_betas_values).hessian
        return FunctionOutput(
            function=bhhh_result.function,
            gradient=bhhh_result.gradient,
            hessian=hessian,
            bhhh=bhhh_result.bhhh,
        )

    def _evaluate_finite_difference_hessian(self, free_betas_values):
        import scipy.optimize as so

        def func_for_fd(betas_array):
            return float(
                self.sum_function(
                    betas_array,
                    self.data_jax,
                    self.draws_jax,
                    self.random_variables_jax,
                )[0]
            )

        eps = np.sqrt(np.finfo(float).eps)
        n = len(free_betas_values)
        the_hessian = np.zeros((n, n), dtype=NUMPY_FLOAT)
        for i in range(n):
            x0 = np.array(free_betas_values)
            ei = np.zeros_like(x0)
            ei[i] = eps
            g_plus = so.approx_fprime(x0 + ei, func_for_fd, eps)
            g_minus = so.approx_fprime(x0 - ei, func_for_fd, eps)
            the_hessian[i, :] = (g_plus - g_minus) / (2 * eps)
        return the_hessian

    def evaluate_individual(
        self,
        the_betas: dict[str, float],
    ) -> np.ndarray:
        """
        Evaluates the compiled expression using provided beta values and returns
        the value of the expression for each observation.

        :param the_betas: Dictionary of parameter names to values.
        :return: A numpy array with one value per observation.
        """
        free_betas_values = (
            self.model_elements.expressions_registry.get_complete_betas_array(
                betas_dict=the_betas
            )
        )

        _, individual_values = self.sum_function(
            free_betas_values, self.data_jax, self.draws_jax, self.random_variables_jax
        )
        return np.asarray(individual_values, dtype=NUMPY_FLOAT)


def calculate_single_formula(
    model_elements: ModelElements,
    the_betas: dict[str, float],
    gradient: bool,
    hessian: bool,
    bhhh: bool,
) -> FunctionOutput:
    """
    Evaluates a single Biogeme expression using JAX, optionally computing the gradient and Hessian.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary of parameter names to values.
    :param gradient: If True, compute the gradient.
    :param hessian: If True, compute the Hessian (requires gradient=True).
    :param bhhh: Unused here, included for compatibility.
    :return: A BiogemeFunctionOutput with the value, gradient, and optionally the Hessian.
    """
    the_compiled_formula = CompiledFormulaEvaluator(model_elements=model_elements)
    return the_compiled_formula.evaluate(
        the_betas=the_betas, gradient=gradient, hessian=hessian, bhhh=bhhh
    )


def calculate_single_formula_from_expression(
    expression: Expression,
    database: Database,
    number_of_draws: int,
    the_betas: dict[str, float],
) -> float:
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        database=database,
        number_of_draws=number_of_draws,
    )
    result = calculate_single_formula(
        model_elements=model_elements,
        the_betas=the_betas,
        gradient=False,
        hessian=False,
        bhhh=False,
    )
    return result.function


def evaluate_formula(
    model_elements: ModelElements,
    the_betas: dict[str, float],
) -> float:
    """
    Evaluates a single Biogeme expression using JAX.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary of parameter names to values.
    :return: the value of the expression.
    """
    result = calculate_single_formula(
        model_elements=model_elements,
        the_betas=the_betas,
        gradient=False,
        hessian=False,
        bhhh=False,
    )
    return result.function


def evaluate_expression_per_row(
    model_elements: ModelElements,
    the_betas: dict[str, float],
) -> np.ndarray:
    """
    Evaluates a Biogeme expression for each entry in the database and returns individual results.

    This function compiles the expression using JAX, applies it to all observations
    in the database, and returns a NumPy array containing the evaluated values
    per observation. The result is not aggregated or summed.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary mapping parameter names to their values.
    :return: A NumPy array of values, one for each observation in the database.
    """
    the_compiled_formula = CompiledFormulaEvaluator(
        model_elements=model_elements,
    )
    return the_compiled_formula.evaluate_individual(the_betas=the_betas)
