"""

Specification of the structural equations
=========================================

Structural equations as functions of the latent variables.

Michel Bierlaire
Wed Sept 03 2025, 08:13:37
"""

from biogeme.expressions import (
    Beta,
    Draws,
    LinearTermTuple,
    LinearUtility,
    Variable,
)

# %%
# Latent variable for the car centric attitude

car_explanatory_variables: list[str] = [
    'highEducation',
    'top_manager',
    'employees',
    'age_30_less',
    'ScaledIncome',
    'car_oriented_parents',
]
car_prefix = 'struct_car'

car_struct_coefficients = {
    variable_name: Beta(f'{car_prefix}_{variable_name}', 0.0, None, None, 0)
    for variable_name in car_explanatory_variables
}
sigma_car_centric = Beta(f'{car_prefix}_sigma', 1.0, None, None, 0)

car_centric_attitude = LinearUtility(
    [
        LinearTermTuple(
            beta=car_struct_coefficients[variable_name], x=Variable(variable_name)
        )
        for variable_name in car_explanatory_variables
    ]
) + sigma_car_centric * Draws(f'{car_prefix}_error_term', 'NORMAL_MLHS_ANTI')

# %%
# Latent variable for the urban preference attitude
# Structural equation

# %%
# Latent variable for the urban preference attitude
urban_explanatory_variables: list[str] = [
    'childSuburb',
    'age_30_less',
    'haveChildren',
    'individualHouse',
    'owningHouse',
    'single',
    'ScaledIncome',
    'city_center_as_kid',
]

urban_prefix = 'struct_urban'
urban_struct_coefficients = {
    variable_name: Beta(f'{urban_prefix}_{variable_name}', 0.0, None, None, 0)
    for variable_name in urban_explanatory_variables
}
sigma_urban_preference = Beta(f'{urban_prefix}_sigma', 1.0, None, None, 0)
urban_preference_attitude = LinearUtility(
    [
        LinearTermTuple(
            beta=urban_struct_coefficients[variable_name], x=Variable(variable_name)
        )
        for variable_name in urban_explanatory_variables
    ]
) + sigma_urban_preference * Draws(f'{urban_prefix}_error_term', 'NORMAL_MLHS_ANTI')
