"""

Relevant data for the hybrid choice model
=========================================

This file identifies the relevant data for the hybrid choice model, that are shared by
several specifications.

Michel Bierlaire, EPFL
Thu May 15 2025, 15:47:42
"""

# %%
# Latent variables
car_centric_name = 'car_centric_attitude'
urban_preference_name = 'urban_preference_attitude'

# %%
# Indicators for the car centric attitude.

car_likert_indicators = {
    'Envir01',
    'Envir02',
    'Envir06',
    'Mobil03',
    'Mobil04',
    'Mobil05',
    'Mobil06',
    'Mobil07',
    'Mobil08',
    'Mobil09',
    'Mobil10',
    'LifSty07',
    'LifSty08',
}
normalized_car = 'Envir01'

# car_binary_indicators = {'moreThanOneCar', 'moreThanOneBike', 'haveGA'}
car_binary_indicators = set()

# %%
# indicators for the urban preference attitude
urban_likert_indicators = {
    'ResidCh01',
    'ResidCh02',
    'ResidCh03',
    'ResidCh05',
    'ResidCh06',
    'ResidCh07',
    'LifSty07',
}
normalized_urban = 'ResidCh01'

# urban_binary_indicators = {'individualHouse'}
urban_binary_indicators = set()


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


normalized = {'Envir01': -1, 'ResidCh01': 1}

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

# %%
# List of all explanatory variables
all_explanatory_variables = car_explanatory_variables + urban_explanatory_variables

# %%
# All indicators
latent_variables_indicators = {
    car_centric_name: car_likert_indicators,
    urban_preference_name: urban_likert_indicators,
}
