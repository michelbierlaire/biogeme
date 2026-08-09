import logging

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from biogeme.default_parameters import ParameterValue
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.optimization import OptimizationAlgorithm
from biogeme.second_derivatives import SecondDerivativesMode
from .model_estimation import AlgorithmResults, model_estimation

logger = logging.getLogger(__name__)


def bootstrap(
    number_of_bootstrap_samples,
    the_algorithm: OptimizationAlgorithm,
    modeling_elements: ModelElements,
    parameters: dict[str, ParameterValue],
    starting_values: dict[str, float],
    second_derivatives_mode: SecondDerivativesMode,
    numerically_safe: bool,
    use_jit: bool,
    number_of_jobs: int,
    analytical_hessian_mode: str = 'full',
    hessian_parameter_block_size: int = 4,
    hessian_observation_batch_size: int = 100,
) -> list[AlgorithmResults]:
    """
    Perform bootstrap estimation to assess the variability of model parameters.

    This function generates a specified number of bootstrap samples from the
    original dataset, estimates the model on each sample using the provided
    algorithm and parameters, and returns the collection of estimation results.

    :param number_of_bootstrap_samples: Number of bootstrap replications to perform.
    :param the_algorithm: The optimization algorithm used to estimate the model.
    :param modeling_elements: The components defining the model, including the database and log-likelihood expression.
    :param parameters: Configuration parameters used during estimation.
    :param starting_values: Dictionary of initial values for the model's free parameters.
    :param second_derivatives_mode: specifies how second derivatives are calculated.
    :param numerically_safe: improves the numerical stability of the calculations.
    :param use_jit: if True, performs just-in-time compilation.
    :param number_of_jobs: number of jobs for parallel execution of bootstrapping.

    :return: A list of tuples containing:
        - estimated parameter values (NumPy array),
        - diagnostic information from the optimizer (dictionary),
        - convergence status (boolean).
    """
    the_database = modeling_elements.database

    def run_one_bootstrap_estimation(_):
        bootstrap_modeling_elements = ModelElements(
            expressions=modeling_elements.expressions,
            adapter=RegularAdapter(database=the_database.bootstrap_sample()),
            number_of_draws=modeling_elements.number_of_draws,
            draws_management=None,
            user_defined_draws=modeling_elements.user_defined_draws,
            expressions_registry=None,
            use_jit=use_jit,
        )
        compiled_formula = CompiledFormulaEvaluator(
            model_elements=bootstrap_modeling_elements,
            second_derivatives_mode=second_derivatives_mode,
            numerically_safe=numerically_safe,
            analytical_hessian_mode=analytical_hessian_mode,
            hessian_parameter_block_size=hessian_parameter_block_size,
            hessian_observation_batch_size=hessian_observation_batch_size,
        )
        one_result = model_estimation(
            the_algorithm=the_algorithm,
            function_evaluator=compiled_formula,
            parameters=parameters,
            some_starting_values=starting_values,
            save_iterations_filename=None,
        )
        return one_result

    PARALLEL = True

    logger.info(f'Number of jobs for bootstrapping: {number_of_jobs}')
    if PARALLEL:
        try:
            with tqdm_joblib(
                tqdm(
                    desc='Bootstraps',
                    total=number_of_bootstrap_samples,
                )
            ):
                results = Parallel(n_jobs=number_of_jobs)(
                    delayed(run_one_bootstrap_estimation)(_)
                    for _ in range(number_of_bootstrap_samples)
                )
            return results
        except (NotImplementedError, PermissionError, ValueError) as exc:
            # Some restricted environments (for example, sandboxed CI jobs)
            # cannot initialize joblib's process backend. The bootstrap itself
            # does not require process workers, so continue serially when the
            # backend cannot be started.
            logger.warning(
                'Parallel bootstrapping is unavailable (%s); continuing serially.',
                exc,
            )

    results = []
    for _ in tqdm(range(number_of_bootstrap_samples), desc='Bootstraps'):
        results.append(run_one_bootstrap_estimation(_))
    return results
