import logging
from dataclasses import dataclass

import pandas as pd
import pymc as pm
from biogeme.bayesian_estimation import (
    BayesianResults,
    RawBayesianResults,
    SamplingConfig,
    run_sampling,
)
from biogeme.default_parameters import ParameterValue
from biogeme.jax_calculator import CompiledFormulaEvaluator, MultiRowEvaluator
from biogeme.likelihood import AlgorithmResults, model_estimation
from biogeme.model_elements import ModelElements
from biogeme.optimization import OptimizationAlgorithm
from biogeme.pymc_calculator import pymc_formula_evaluator

from .split_databases import EstimationValidationModels, split_databases

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    estimation_modeling_elements: ModelElements
    validation_modeling_elements: ModelElements
    simulated_values: pd.DataFrame


def cross_validate_model(
    the_algorithm: OptimizationAlgorithm,
    modeling_elements: ModelElements,
    parameters: dict[str, ParameterValue],
    starting_values: dict[str, float],
    slices: int,
    numerically_safe: bool,
    groups: str | None = None,
) -> list[ValidationResult]:
    validation_models: list[EstimationValidationModels] = split_databases(
        model_elements=modeling_elements, slices=slices, groups=groups
    )
    results = []
    for i, fold in enumerate(validation_models, 1):
        # Estimation phase
        the_function_evaluator = CompiledFormulaEvaluator(
            model_elements=fold.estimation,
            second_derivatives_mode=parameters['calculating_second_derivatives'],
            numerically_safe=numerically_safe,
        )
        one_result: AlgorithmResults = model_estimation(
            the_algorithm=the_algorithm,
            function_evaluator=the_function_evaluator,
            parameters=parameters,
            some_starting_values=starting_values,
            save_iterations_filename=None,
        )
        estimated_betas = fold.estimation.expressions_registry.get_named_betas_values(
            values=one_result.solution
        )
        simulation_evaluator = MultiRowEvaluator(
            model_elements=fold.validation,
            numerically_safe=numerically_safe,
            use_jit=modeling_elements.use_jit,
        )
        simulated_values: pd.DataFrame = simulation_evaluator.evaluate(
            the_betas=estimated_betas
        )
        result = ValidationResult(
            estimation_modeling_elements=fold.estimation,
            validation_modeling_elements=fold.validation,
            simulated_values=simulated_values,
        )
        results.append(result)

    return results


def bayesian_cross_validate_model(
    sampling_config: SamplingConfig,
    modeling_elements: ModelElements,
    parameters: dict[str, ParameterValue],
    starting_values: dict[str, float],
    slices: int,
    groups: str | None = None,
) -> list[ValidationResult]:
    validation_models: list[EstimationValidationModels] = split_databases(
        model_elements=modeling_elements, slices=slices, groups=groups
    )
    results = []
    for i, fold in enumerate(validation_models, 1):
        model_name = f'validation_{i}'
        # Estimation phase
        with pm.Model() as model:
            loglike_total = pymc_formula_evaluator(model_elements=modeling_elements)
            pm.Deterministic(modeling_elements.loglikelihood_name, loglike_total)
            pm.Potential("choice_logp", loglike_total)

            idata, used_numpyro = run_sampling(
                model=model,
                draws=parameters['bayesian_draws'],
                tune=parameters['warmup'],
                chains=parameters['chains'],
                config=sampling_config,
            )
        bayes_results = RawBayesianResults(
            idata=idata,
            model_name=model_name,
            data_name=modeling_elements.database.name,
            beta_names=modeling_elements.free_betas_names,
            sampler='NUTS',
            target_accept=parameters['target_accept'],
        )
        one_result = BayesianResults(raw=bayes_results)
        estimated_betas = one_result.get_beta_values()
        simulation_evaluator = MultiRowEvaluator(
            model_elements=fold.validation,
            numerically_safe=True,
            use_jit=modeling_elements.use_jit,
        )
        simulated_values: pd.DataFrame = simulation_evaluator.evaluate(
            the_betas=estimated_betas
        )
        result = ValidationResult(
            estimation_modeling_elements=fold.estimation,
            validation_modeling_elements=fold.validation,
            simulated_values=simulated_values,
        )
        results.append(result)

    return results
