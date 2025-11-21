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


normalized_car = 'Envir01'

# car_binary_indicators = {'moreThanOneCar', 'moreThanOneBike', 'haveGA'}
car_binary_indicators = set()


normalized_urban = 'ResidCh01'

# urban_binary_indicators = {'individualHouse'}
urban_binary_indicators = set()


normalized = {'Envir01': -1, 'ResidCh01': 1}


# %%
# List of all explanatory variables
all_explanatory_variables = car_explanatory_variables + urban_explanatory_variables

# %%
# All indicators
latent_variables_indicators = {
    car_centric_name: car_likert_indicators,
    urban_preference_name: urban_likert_indicators,
}
