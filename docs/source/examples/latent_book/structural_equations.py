from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Expression, MultipleSum
from relevant_data import car_explanatory_variables, urban_explanatory_variables


# %%
# Structural equation: car centric attitude
def build_car_centric_attitude(
    with_beta_parameters: bool = True,
    estimated_parameters: dict[str, float] | None = None,
) -> Expression:
    """Builds the expression for the structural equation of the car centric attitude

    :param with_beta_parameters: if True, parameters have to be estimated.
    :param estimated_parameters: if not None, provides the value of the parameters, either for the direct
        calculation (with_beta_parameters == False) or for starting value (with_beta_parameters == True)
    :return: the expression for the structural equation
    """
    if estimated_parameters is None:
        if not with_beta_parameters:
            raise BiogemeError('Values for the parameters must be provided.')
        car_struct_coefficients = {
            variable_name: Beta(f'car_struct_{variable_name}', 0.0, None, None, 0)
            for variable_name in car_explanatory_variables.keys()
        }
        car_struct_intercept = Beta('car_struct_intercept', 0.0, None, None, 0)
    elif with_beta_parameters:
        car_struct_coefficients = {
            variable_name: Beta(
                f'car_struct_{variable_name}',
                estimated_parameters[f'car_struct_{variable_name}'],
                None,
                None,
                0,
            )
            for variable_name in car_explanatory_variables.keys()
        }
        car_struct_intercept = Beta(
            'car_struct_intercept',
            estimated_parameters['car_struct_intercept'],
            None,
            None,
            0,
        )
    else:
        car_struct_coefficients = {
            variable_name: estimated_parameters[f'car_struct_{variable_name}']
            for variable_name in car_explanatory_variables.keys()
        }
        car_struct_intercept = estimated_parameters['car_struct_intercept']

    car_centric_attitude = car_struct_intercept + MultipleSum(
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
    with_beta_parameters: bool = True,
    estimated_parameters: dict[str, float] | None = None,
) -> Expression:
    """Builds the expression for the structural equation of the urban preference attitude

    :param with_beta_parameters: if True, parameters have to be estimated.
    :param estimated_parameters: if not None, provides the value of the parameters, either for the direct
        calculation (with_beta_parameters == False) or for starting value (with_beta_parameters == True)
    :return: the expression for the structural equation
    """
    if estimated_parameters is None:
        if not with_beta_parameters:
            raise BiogemeError('Values for the parameters must be provided.')
        urban_struct_coefficients = {
            variable_name: Beta(f'urban_struct_{variable_name}', 0.0, None, None, 0)
            for variable_name in urban_explanatory_variables.keys()
        }
        urban_struct_intercept = Beta('urban_struct_intercept', 0.0, None, None, 0)
    elif with_beta_parameters:
        urban_struct_coefficients = {
            variable_name: Beta(
                f'urban_struct_{variable_name}',
                estimated_parameters[f'urban_struct_{variable_name}'],
                None,
                None,
                0,
            )
            for variable_name in urban_explanatory_variables.keys()
        }
        urban_struct_intercept = Beta(
            'urban_struct_intercept',
            estimated_parameters['urban_struct_intercept'],
            None,
            None,
            0,
        )

    else:
        urban_struct_coefficients = {
            variable_name: estimated_parameters[f'urban_struct_{variable_name}']
            for variable_name in urban_explanatory_variables.keys()
        }
        urban_struct_intercept = estimated_parameters['urban_struct_intercept']

    urban_life_attitude = urban_struct_intercept + MultipleSum(
        [
            urban_struct_coefficients[variable_name] * variable_expression
            for variable_name, variable_expression in urban_explanatory_variables.items()
        ]
    )
    return urban_life_attitude
