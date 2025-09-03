"""

Specification of the structural equations
=========================================

Structural equations as functions of the latent variables.

Michel Bierlaire
Wed Sept 03 2025, 08:13:37
"""

from typing import NamedTuple

from relevant_data import (
    car_centric_name,
    car_explanatory_variables,
    urban_explanatory_variables,
    urban_preference_name,
)

from biogeme.expressions import (
    Beta,
    Draws,
    Expression,
    LinearTermTuple,
    LinearUtility,
    Variable,
)


class LatentVariable(NamedTuple):
    name: str
    expression: Expression


# %%
# Structural equation: car centric attitude
def build_car_centric_attitude(
    estimated_parameters: dict[str, float] | None = None,
) -> LatentVariable:
    """Builds the expression for the structural equation of the car centric attitude

    :param estimated_parameters: if not None, provides the value of the parameters for
        the direct calculation.
    :return: the expression for the structural equation
    """
    prefix = 'struct_car'
    if estimated_parameters is None:
        car_struct_coefficients = {
            variable_name: Beta(f'{prefix}_{variable_name}', 0.0, None, None, 0)
            for variable_name in car_explanatory_variables
        }
        sigma_car_centric = Beta(f'{prefix}_sigma', 1.0, None, None, 0)

    else:
        car_struct_coefficients = {
            variable_name: estimated_parameters[f'{prefix}_{variable_name}']
            for variable_name in car_explanatory_variables
        }
        sigma_car_centric = estimated_parameters[f'{prefix}_sigma']

    car_centric_attitude = LinearUtility(
        [
            LinearTermTuple(
                beta=car_struct_coefficients[variable_name], x=Variable(variable_name)
            )
            for variable_name in car_explanatory_variables
        ]
    ) + sigma_car_centric * Draws(f'{prefix}_error_term', 'NORMAL_MLHS_ANTI')

    return LatentVariable(name=car_centric_name, expression=car_centric_attitude)


# %%
# Latent variable for the urban preference attitude
# Structural equation
def build_urban_preference_attitude(
    estimated_parameters: dict[str, float] | None = None,
) -> LatentVariable:
    """Builds the expression for the structural equation of the urban preference
        attitude

    :param estimated_parameters: if not None, provides the value of the parameters for
        the direct calculation
    :return: the expression for the structural equation
    """
    prefix = 'struct_urban'

    if estimated_parameters is None:
        urban_struct_coefficients = {
            variable_name: Beta(f'{prefix}_{variable_name}', 0.0, None, None, 0)
            for variable_name in urban_explanatory_variables
        }
        sigma_urban_preference = Beta(f'{prefix}_sigma', 1.0, None, None, 0)
    else:
        urban_struct_coefficients = {
            variable_name: estimated_parameters[f'{prefix}_{variable_name}']
            for variable_name in urban_explanatory_variables
        }
        sigma_urban_preference = estimated_parameters[f'{prefix}_sigma']

    urban_preference_attitude = LinearUtility(
        [
            LinearTermTuple(
                beta=urban_struct_coefficients[variable_name], x=Variable(variable_name)
            )
            for variable_name in urban_explanatory_variables
        ]
    ) + sigma_urban_preference * Draws(f'{prefix}_error_term', 'NORMAL_MLHS_ANTI')
    return LatentVariable(
        name=urban_preference_name, expression=urban_preference_attitude
    )
