from relevant_data import car_explanatory_variables, urban_explanatory_variables

from biogeme.expressions import Beta, Expression, MultipleSum


# %%
# Structural equation: car centric attitude
def build_car_centric_attitude(
    estimated_parameters: dict[str, float] | None = None,
) -> Expression:
    """Builds the expression for the structural equation of the car centric attitude

    :param estimated_parameters: if not None, provides the value of the parameters for
        the direct calculation.
    :return: the expression for the structural equation
    """
    if estimated_parameters is None:
        car_struct_coefficients =  {
            variable_name: Beta(f'car_struct_{variable_name}', 0.0, None, None, 0)
            for variable_name in car_explanatory_variables.keys()
        }
    else:
        car_struct_coefficients = {
            variable_name: estimated_parameters[f'car_struct_{variable_name}']
            for variable_name in car_explanatory_variables.keys()
        }

    car_centric_attitude = MultipleSum(
        [
            car_struct_coefficients[variable_name] * variable_expression
            for variable_name, variable_expression in car_explanatory_variables.items()
        ]
    )
    return car_centric_attitude


# %%
# Latent variable for the urban preference attitude
# Structural equation
def build_urban_preference_attitude(
    estimated_parameters: dict[str, float] | None = None,
) -> Expression:
    """Builds the expression for the structural equation of the urban preference
        attitude

    :param estimated_parameters: if not None, provides the value of the parameters for
        the direct calculation
    :return: the expression for the structural equation
    """
    if estimated_parameters is None:
        urban_struct_coefficients = {
            variable_name: Beta(f'urban_struct_{variable_name}', 0.0, None, None, 0)
            for variable_name in urban_explanatory_variables.keys()
        }
    else:
        urban_struct_coefficients = {
            variable_name: estimated_parameters[f'urban_struct_{variable_name}']
            for variable_name in urban_explanatory_variables.keys()
        }

    urban_life_attitude = MultipleSum(
        [
            urban_struct_coefficients[variable_name] * variable_expression
            for variable_name, variable_expression in urban_explanatory_variables.items()
        ]
    )
    return urban_life_attitude
