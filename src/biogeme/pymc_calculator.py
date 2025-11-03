"""
Module in charge of the actual calculation of the formula on the database for Bayesian estination.

Michel Bierlaire
Tue Oct 28 2025, 11:45:59
"""

from __future__ import annotations

import logging

import pytensor.tensor as pt
from biogeme.exceptions import BiogemeError
from biogeme.model_elements import ModelElements

logger = logging.getLogger(__name__)


def pymc_formula_evaluator(model_elements: ModelElements) -> pt.TensorVariable:
    """
    Prepares and compiles the PyMc function for evaluating a Biogeme expression.

    :param model_elements: All elements needed to calculate the expression.
    """

    log_likelihood = model_elements.loglikelihood
    if log_likelihood is None:
        error_message = (
            f'No expression found for log likelihood. '
            f'Available expressions: {model_elements.formula_names}'
        )
        raise BiogemeError(error_message)
    pymc_builder = log_likelihood.recursive_construct_pymc_model_builder()
    pymc_expression = pymc_builder(dataframe=model_elements.database.dataframe)

    if model_elements.weight is not None:
        weight_builder = model_elements.weight.recursive_construct_pymc_model_builder()
        weight_expression = weight_builder(dataframe=model_elements.database.dataframe)
        total_expression = weight_expression * pymc_expression
    else:
        total_expression = pymc_expression

    return total_expression
