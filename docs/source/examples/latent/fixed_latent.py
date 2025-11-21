"""

Specification of the structural equations
=========================================

Structural equations as functions of the latent variables.

Michel Bierlaire
Wed Sept 03 2025, 08:13:37
"""

from biogeme.expressions import (
    Beta,
    LinearTermTuple,
    LinearUtility,
    Variable,
)

structural_parameters = {
    "struct_car_highEducation": Beta(
        "struct_car_highEducation", -0.9122371636946164, None, None, 1
    ),
    "struct_car_top_manager": Beta(
        "struct_car_top_manager", 0.30846466969640945, None, None, 1
    ),
    "struct_car_employees": Beta(
        "struct_car_employees", -0.00034793593098248035, None, None, 1
    ),
    "struct_car_age_30_less": Beta(
        "struct_car_age_30_less", 0.6112537854181949, None, None, 1
    ),
    "struct_car_ScaledIncome": Beta(
        "struct_car_ScaledIncome", -0.01871430165561051, None, None, 1
    ),
    "struct_car_car_oriented_parents": Beta(
        "struct_car_car_oriented_parents", 0.49707082930953034, None, None, 1
    ),
    "struct_car_sigma": Beta("struct_car_sigma", 1.5026868723209956, None, None, 1),
    "choice_urban_life_pt_cte": Beta(
        "choice_urban_life_pt_cte", 6.658354053859266, None, None, 1
    ),
    "struct_urban_childSuburb": Beta(
        "struct_urban_childSuburb", 0.19019612391050283, None, None, 1
    ),
    "struct_urban_age_30_less": Beta(
        "struct_urban_age_30_less", 0.3756227502880287, None, None, 1
    ),
    "struct_urban_haveChildren": Beta(
        "struct_urban_haveChildren", 0.029389519219771204, None, None, 1
    ),
    "struct_urban_individualHouse": Beta(
        "struct_urban_individualHouse", -0.18772990463421932, None, None, 1
    ),
    "struct_urban_owningHouse": Beta(
        "struct_urban_owningHouse", 0.17285359144629064, None, None, 1
    ),
    "struct_urban_single": Beta(
        "struct_urban_single", -0.09211879307017229, None, None, 1
    ),
    "struct_urban_ScaledIncome": Beta(
        "struct_urban_ScaledIncome", 0.01421489863656365, None, None, 1
    ),
    "struct_urban_city_center_as_kid": Beta(
        "struct_urban_city_center_as_kid", 0.36602227749264854, None, None, 1
    ),
    "struct_urban_sigma": Beta("struct_urban_sigma", 0.5491435168760334, None, None, 1),
}
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
    variable_name: structural_parameters[f'{car_prefix}_{variable_name}']
    for variable_name in car_explanatory_variables
}

car_centric_attitude = LinearUtility(
    [
        LinearTermTuple(
            beta=car_struct_coefficients[variable_name], x=Variable(variable_name)
        )
        for variable_name in car_explanatory_variables
    ]
)

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
    variable_name: structural_parameters[f'{urban_prefix}_{variable_name}']
    for variable_name in urban_explanatory_variables
}
urban_preference_attitude = LinearUtility(
    [
        LinearTermTuple(
            beta=urban_struct_coefficients[variable_name], x=Variable(variable_name)
        )
        for variable_name in urban_explanatory_variables
    ]
)
