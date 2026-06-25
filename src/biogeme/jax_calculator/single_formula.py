"""
Module in charge of the actual calculation of the formula on the database.

Michel Bierlaire
Wed Mar 26 19:30:57 2025
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    collect_init_values,
)
from biogeme.floating_point import JAX_FLOAT, NUMPY_FLOAT
from biogeme.function_output import FunctionOutput, NamedFunctionOutput
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.profiling import JaxExecutionProfile
from biogeme.second_derivatives import SecondDerivativesMode

logger = logging.getLogger(__name__)


class CompiledFormulaEvaluator:
    """
    Compiles and evaluates a Biogeme expression using JAX for efficient
        repeated computation.
    """

    def __init__(
        self,
        model_elements: ModelElements,
        second_derivatives_mode: SecondDerivativesMode,
        numerically_safe: bool,
        profiler: JaxExecutionProfile | None = None,
    ):
        """
        Prepares and compiles the JAX function for evaluating a Biogeme expression.

        :param model_elements: All elements needed to calculate the expression.
        :param second_derivatives_mode: specifies how second derivatives are calculated.
        :param numerically_safe: improves the numerical stability of the calculations.
        :param profiler: optional execution profiler used to record build counts,
            call counts, signatures, and timings of JAX-related functions.
        """
        from biogeme.expressions import build_vectorized_function

        self.model_elements = model_elements
        self.second_derivatives_mode = second_derivatives_mode
        self.numerically_safe = numerically_safe
        self.use_jit = model_elements.use_jit
        self.profiler = profiler if profiler is not None else JaxExecutionProfile()
        self.free_betas_names = (
            self.model_elements.expressions_registry.free_betas_names
        )
        self.data_jax = (
            self.model_elements.database.data_jax
            if self.model_elements.database is not None
            else None
        )
        self.draws_jax = (
            self.model_elements.draws_management.draws_jax
            if self.model_elements.draws_management is not None
            else None
        )
        n_rv = self.model_elements.expressions_registry.number_of_random_variables
        self.random_variables_jax = jnp.zeros((n_rv,), dtype=JAX_FLOAT)

        log_likelihood = self.model_elements.loglikelihood
        if log_likelihood is None:
            error_message = (
                f'No expression found for log likelihood. '
                f'Available expressions: {self.model_elements.formula_names}'
            )
            raise BiogemeError(error_message)
        self.row_loglikelihood_function = (
            log_likelihood.recursive_construct_jax_function(
                numerically_safe=self.numerically_safe
            )
        )
        self.profiler.record_build('row_loglikelihood_function')
        self.vectorized_loglikelihood_function = build_vectorized_function(
            self.row_loglikelihood_function,
            use_jit=self.use_jit,
            profiler=self.profiler,
            profile_name='vectorized_loglikelihood_function',
        )

        if self.model_elements.weight is not None:
            weight_function = (
                self.model_elements.weight.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                )
            )
            self.vectorized_weight_function = build_vectorized_function(
                weight_function,
                use_jit=self.use_jit,
                profiler=self.profiler,
                profile_name='vectorized_weight_function',
            )
        else:
            self.vectorized_weight_function = None

        def sum_function(
            params: list[float],
            data: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            values = self.vectorized_loglikelihood_function(
                params, data, draws, random_variables
            )
            if self.vectorized_weight_function is not None:
                weights = self.vectorized_weight_function(
                    params, data, draws, random_variables
                )
                values *= weights
            return jnp.asarray(jnp.sum(values), dtype=JAX_FLOAT), values

        self.sum_function = jax.jit(sum_function) if self.use_jit else sum_function
        self.profiler.record_build('sum_function')

        def scalar_function(
            params: list[float],
            data: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            return self.sum_function(params, data, draws, random_variables)[0]

        self.scalar_function = (
            jax.jit(scalar_function) if self.use_jit else scalar_function
        )
        self.profiler.record_build('scalar_function')
        self.value_and_grad_function = jax.value_and_grad(
            self.scalar_function,
            argnums=0,
        )
        self.profiler.record_build('value_and_grad_function')
        self.autodiff_hessian_function = jax.jacfwd(
            jax.grad(self.scalar_function, argnums=0),
            argnums=0,
        )
        self.profiler.record_build('autodiff_hessian_function')

        def one_observation_loglikelihood(
            params: list[float],
            row: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            # Important: this is intentionally the *unweighted* contribution of one
            # observation. For the BHHH matrix, the observation weight must multiply
            # each outer-product contribution only once. If the weight were applied
            # here, before differentiation, the resulting gradient would already be
            # scaled by the weight and the outer product would therefore include the
            # weight squared, which is incorrect.
            #
            # Also important: the row-level JAX function expects the complete draw
            # array for one observation. In particular, MonteCarlo expressions perform
            # their own internal vectorization/integration over draws. Therefore we
            # must pass the full `draws` array directly here, and not vectorize over
            # draws again. This keeps the semantics identical to the previous code
            # while avoiding routing a single observation through the full batch-
            # oriented vectorized wrapper.
            return self.row_loglikelihood_function(params, row, draws, random_variables)

        per_obs_value_and_grad_fn = jax.vmap(
            jax.value_and_grad(one_observation_loglikelihood, argnums=0),
            in_axes=(None, 0, 0, None),
        )
        self.per_observation_value_and_grad_function = (
            jax.jit(per_obs_value_and_grad_fn)
            if self.use_jit
            else per_obs_value_and_grad_fn
        )
        self.profiler.record_build('per_observation_value_and_grad_function')

        def bhhh_function(
            params: jnp.ndarray,
            data: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            individual_values, individual_gradients = per_obs_value_and_grad_fn(
                params,
                data,
                draws,
                random_variables,
            )

            if self.vectorized_weight_function is not None:
                individual_weights = self.vectorized_weight_function(
                    params,
                    data,
                    draws,
                    random_variables,
                )
            else:
                individual_weights = jnp.ones((data.shape[0],), dtype=JAX_FLOAT)

            weighted_values = individual_values * individual_weights
            weighted_individual_gradients = (
                individual_gradients * individual_weights[:, None]
            )
            the_gradient = jnp.sum(weighted_individual_gradients, axis=0)
            bhhh_matrix = individual_gradients.T @ weighted_individual_gradients

            return jnp.sum(weighted_values), the_gradient, bhhh_matrix

        self.bhhh_function = jax.jit(bhhh_function) if self.use_jit else bhhh_function
        self.profiler.record_build('bhhh_function')

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
        value_jax, _ = self.profiler.timed_call(
            'sum_function',
            self.sum_function,
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
        value, the_gradient = self.profiler.timed_call(
            'value_and_grad_function',
            self.value_and_grad_function,
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
        if self.second_derivatives_mode == SecondDerivativesMode.NEVER:
            error_msg = 'The second derivatives are not supposed to be evaluated'
            raise BiogemeError(error_msg)

        value, the_gradient = self.profiler.timed_call(
            'value_and_grad_function',
            self.value_and_grad_function,
            free_betas_values,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )
        if jnp.all(the_gradient == 0.0):
            the_hessian = np.zeros(
                (len(free_betas_values), len(free_betas_values)), dtype=NUMPY_FLOAT
            )
        elif self.second_derivatives_mode == SecondDerivativesMode.FINITE_DIFFERENCES:
            the_hessian = self._evaluate_finite_difference_hessian(free_betas_values)
        else:
            hess_autodiff = self.profiler.timed_call(
                'autodiff_hessian_function',
                self.autodiff_hessian_function,
                free_betas_values,
                self.data_jax,
                self.draws_jax,
                self.random_variables_jax,
            )
            if jnp.any(jnp.isnan(hess_autodiff)):
                logger.warning(
                    'The calculation of second derivatives generated numerical errors.'
                )
            #    raise BiogemeError(
            #        'The calculation of second derivatives generated numerical errors.'
            #    )

            the_hessian = np.asarray(hess_autodiff, dtype=NUMPY_FLOAT)

        return FunctionOutput(
            function=float(value),
            gradient=np.asarray(the_gradient, dtype=NUMPY_FLOAT),
            hessian=the_hessian,
            bhhh=None,
        )

    def _evaluate_bhhh_only(self, free_betas_values):
        free_betas_values_jnp = jnp.asarray(free_betas_values, dtype=JAX_FLOAT)
        value, the_gradient, bhhh_matrix = self.profiler.timed_call(
            'bhhh_function',
            self.bhhh_function,
            free_betas_values_jnp,
            self.data_jax,
            self.draws_jax,
            self.random_variables_jax,
        )

        expected_shape = (len(free_betas_values), len(free_betas_values))
        if bhhh_matrix.shape != expected_shape:
            error_msg = (
                f'Unexpected shape for BHHH matrix: '
                f'{bhhh_matrix.shape}. Expected {expected_shape}.'
            )
            raise BiogemeError(error_msg)

        return FunctionOutput(
            function=float(value),
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
    second_derivatives_mode: SecondDerivativesMode,
    numerically_safe: bool,
    profiler: JaxExecutionProfile | None = None,
) -> FunctionOutput:
    """
    Evaluates a single Biogeme expression using JAX, optionally computing the gradient
        and Hessian.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary of parameter names to values.
    :param gradient: If True, compute the gradient.
    :param hessian: If True, compute the Hessian (requires gradient=True).
    :param bhhh: Unused here, included for compatibility.
    :param second_derivatives_mode: specifies how second derivatives are calculated.
    :param numerically_safe: improves the numerical stability of the calculations.
    :param profiler: optional execution profiler used to record JAX build counts
        and timings.
    :return: A BiogemeFunctionOutput with the value, gradient,
       and optionally the Hessian.
    """
    the_compiled_formula = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=second_derivatives_mode,
        numerically_safe=numerically_safe,
        profiler=profiler,
    )
    return the_compiled_formula.evaluate(
        the_betas=the_betas, gradient=gradient, hessian=hessian, bhhh=bhhh
    )


def calculate_single_formula_from_expression(
    expression: Expression,
    database: Database,
    number_of_draws: int,
    the_betas: dict[str, float],
    second_derivatives_mode: SecondDerivativesMode,
    numerically_safe: bool,
    use_jit: bool,
) -> float:
    adapter = (
        FlatPanelAdapter(database=database)
        if database.is_panel()
        else RegularAdapter(database=database)
    )
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=adapter,
        number_of_draws=number_of_draws,
        use_jit=use_jit,
    )
    result = calculate_single_formula(
        model_elements=model_elements,
        second_derivatives_mode=second_derivatives_mode,
        numerically_safe=numerically_safe,
        the_betas=the_betas,
        gradient=False,
        hessian=False,
        bhhh=False,
    )
    return result.function


def evaluate_formula(
    model_elements: ModelElements,
    the_betas: dict[str, float],
    second_derivatives_mode: SecondDerivativesMode,
    numerically_safe: bool,
) -> float:
    """
    Evaluates a single Biogeme expression using JAX.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary of parameter names to values.
    :param second_derivatives_mode: specifies how second derivatives are calculated.
    :param numerically_safe: improves the numerical stability of the calculations.
    :return: the value of the expression.
    """
    result = calculate_single_formula(
        model_elements=model_elements,
        the_betas=the_betas,
        gradient=False,
        hessian=False,
        bhhh=False,
        second_derivatives_mode=second_derivatives_mode,
        numerically_safe=numerically_safe,
    )
    return result.function


def evaluate_model_per_row(
    model_elements: ModelElements,
    the_betas: dict[str, float],
    second_derivatives_mode: SecondDerivativesMode,
    numerically_safe: bool,
) -> np.ndarray:
    """
    Evaluates a Biogeme expression for each entry in the database and returns
    individual results.

    This function compiles the expression using JAX, applies it to all observations
    in the database, and returns a NumPy array containing the evaluated values
    per observation. The result is not aggregated or summed.

    :param model_elements: All elements needed to calculate the expression.
    :param the_betas: Dictionary mapping parameter names to their values.
    :param second_derivatives_mode: specifies how second derivatives are calculated.
    :param numerically_safe: improves the numerical stability of the calculations.
    :return: A NumPy array of values, one for each observation in the database.
    """
    the_compiled_formula = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=second_derivatives_mode,
        numerically_safe=numerically_safe,
    )
    return the_compiled_formula.evaluate_individual(the_betas=the_betas)


def evaluate_expression(
    expression: Expression,
    numerically_safe: bool,
    use_jit: bool,
    database: Database | None = None,
    betas: dict[str, float] | None = None,
    number_of_draws: int = 1000,
    aggregation: bool = False,
) -> np.ndarray | float:
    """Evaluate an arithmetic expression

    :param expression: the expression to be evaluated
    :param numerically_safe: if True, the numerical stability of the evaluation is improved, possibly at the expense
         of calculation speed. Set it to False except if necessary.
    :param use_jit: if True, performs just-in-time compilation.
    :param database: database, needed if the expression involves `Variable`
    :param betas: values of the parameters, if the expression involves `Beta`
    :param number_of_draws: number of draws for Monte Carlo integration, if the expression involves it.
    :param aggregation: if True, the sum over all rows is calculated. If False, the value for each row is returned.
    """
    if database is None:
        database = Database.dummy_database()
    adapter = (
        FlatPanelAdapter(database=database)
        if database.is_panel()
        else RegularAdapter(database=database)
    )
    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=adapter,
        number_of_draws=number_of_draws,
        use_jit=use_jit,
    )
    if betas is None:
        betas = collect_init_values(expression=expression)
    if aggregation:
        return evaluate_formula(
            model_elements=model_elements,
            the_betas=betas,
            second_derivatives_mode=SecondDerivativesMode.NEVER,
            numerically_safe=numerically_safe,
        )

    return evaluate_model_per_row(
        model_elements=model_elements,
        the_betas=betas,
        second_derivatives_mode=SecondDerivativesMode.NEVER,
        numerically_safe=numerically_safe,
    )


def get_value_and_derivatives(
    expression: Expression,
    numerically_safe: bool,
    use_jit: bool,
    betas: dict[str, float] | None = None,
    database: Database | None = None,
    number_of_draws: int = 1000,
    gradient: bool = True,
    hessian: bool = True,
    bhhh: bool = True,
    named_results: bool = False,
) -> FunctionOutput | NamedFunctionOutput:
    if database is None:
        from biogeme.database import Database

        database = Database.dummy_database()
    adapter = (
        FlatPanelAdapter(database=database)
        if database.is_panel()
        else RegularAdapter(database=database)
    )

    model_elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=adapter,
        number_of_draws=number_of_draws,
        use_jit=use_jit,
    )

    the_compiled_formula = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )
    if betas is None:
        betas = collect_init_values(expression=expression)
    result: FunctionOutput = the_compiled_formula.evaluate(
        the_betas=betas, gradient=gradient, hessian=hessian, bhhh=bhhh
    )
    if not named_results:
        return result
    named_results = NamedFunctionOutput(
        function_output=result,
        mapping=model_elements.expressions_registry.free_betas_indices,
    )
    return named_results


def get_value_c(
    expression: Expression,
    numerically_safe: bool,
    use_jit: bool,
    database: Database | None = None,
    betas: dict[str, float] | None = None,
    number_of_draws: int = 1000,
    aggregation: bool = False,
) -> np.ndarray | float:
    """For backward compatibility. This function used to be a member of the
    Expression class."""
    return evaluate_expression(
        expression=expression,
        numerically_safe=numerically_safe,
        database=database,
        betas=betas,
        number_of_draws=number_of_draws,
        aggregation=aggregation,
        use_jit=use_jit,
    )
