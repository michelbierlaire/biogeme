from itertools import chain
from typing import NamedTuple

from structural_equations import LatentVariable

from biogeme.expressions import Beta, Expression, MultipleSum, Numeric


class MeasurementEquation(NamedTuple):
    intercept: Expression
    linear_terms: Expression
    scale_parameter: Expression


# %%
# Measurement equations
def generate_continuous_measurement_equations(
    latent_variables: list[LatentVariable],
    latent_variables_indicators: dict[str, list[str]],
    normalized: dict[str, float],
) -> dict[str, MeasurementEquation]:
    all_indicators = set(chain.from_iterable(latent_variables_indicators.values()))
    result_dict = {}
    for indicator in all_indicators:

        # Intercept
        intercept = Beta(f'meas_intercept_{indicator}', 0, None, None, 0)

        # Coefficients for each latent_old variables
        list_of_terms = list()
        for latent_variable in latent_variables:
            list_of_indicators = latent_variables_indicators[latent_variable.name]
            if indicator in list_of_indicators:
                coefficient = (
                    normalized[indicator]
                    if indicator in normalized
                    else Beta(
                        f'meas_{latent_variable.name}_coeff_{indicator}',
                        0,
                        None,
                        None,
                        0,
                    )
                )
                term = coefficient * latent_variable.expression
                list_of_terms.append(term)

        # Scale parameters of the error terms.
        scale_parameter = (
            Numeric(1)
            if indicator in normalized.keys()
            else Beta(f'meas_scale_{indicator}', 1, 1.0e-4, None, 0)
        )
        the_equation = MeasurementEquation(
            intercept=intercept,
            linear_terms=MultipleSum(list_of_terms) if list_of_terms else Numeric(0),
            scale_parameter=scale_parameter,
        )
        result_dict[indicator] = the_equation
    return result_dict
