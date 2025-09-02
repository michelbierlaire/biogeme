"""
Calculates the contributions to the log likelihood of each indicator
"""

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Elem, Expression, MultipleProduct, Numeric, Variable
from biogeme.models import ordered_probit_from_thresholds

from measurement_equations_continuous import MeasurementEquation


def likelihood_discrete_mimic(
    measurement_equations: dict[str, MeasurementEquation],
    threshold_parameters: list[Expression],
    discrete_values: list[int],
    missing_values: list[int],
) -> Expression:
    list_of_factors = []
    for indicator, measurement_equation in measurement_equations.items():
        dict_of_probabilities = ordered_probit_from_thresholds(
            continuous_value=measurement_equation.intercept
            + measurement_equation.linear_terms,
            scale_parameter=measurement_equation.scale_parameter,
            list_of_discrete_values=discrete_values,
            threshold_parameters=threshold_parameters,
        )
        for index in missing_values:
            if index in discrete_values:
                err_msg = (
                    f'Index {index} cannot be both a Likert index and a missing value'
                )
                raise BiogemeError(err_msg)
            dict_of_probabilities[index] = Numeric(1.0)
        list_of_factors.append(Elem(dict_of_probabilities, Variable(indicator)))

    return MultipleProduct(list_of_factors)
