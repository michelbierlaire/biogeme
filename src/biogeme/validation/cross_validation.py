import logging
from dataclasses import dataclass

import pandas as pd

from biogeme.calculator import CompiledFormulaEvaluator, MultiRowEvaluator
from biogeme.default_parameters import ParameterValue
from biogeme.likelihood import AlgorithmResults, model_estimation
from biogeme.model_elements import ModelElements
from biogeme.optimization import OptimizationAlgorithm
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
            avoid_analytical_second_derivatives=parameters[
                'avoid_analytical_second_derivatives'
            ],
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
        simulation_evaluator = MultiRowEvaluator(model_elements=fold.validation)
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
