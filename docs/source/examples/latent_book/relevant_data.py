"""

Relevant data for the hybrid choice model
=========================================

This file identifies the relevant data for the hybrid choice model, that are shared by several specifications.
Michel Bierlaire, EPFL
Thu May 15 2025, 15:47:42
"""

from biogeme.data.optima import (
    HouseType,
    ScaledIncome,
    SocioProfCat,
    UrbRur,
    age,
    age_65_more,
    childCenter,
    childSuburb,
    haveChildren,
    haveGA,
    highEducation,
    individualHouse,
    moreThanOneBike,
    moreThanOneCar,
)
from biogeme.expressions import Expression

# %%
# Indicators for the car centric attitude.

car_indicators = {
    'Envir01',
    'Envir02',
    'Envir03',
    'Envir04',
    'Mobil09',
    'Mobil11',
    'Mobil14',
    'Mobil16',
    'Mobil17',
    'LifSty08',
}
normalized_car = 'Envir01'

# %%
# indicators for the urban preference attitude
urban_indicators = {
    'ResidCh01',
    'ResidCh02',
    'ResidCh03',
    'ResidCh05',
    'ResidCh06',
    'ResidCh07',
    'Mobil07',
    'Mobil24',
}
normalized_urban = 'ResidCh01'

# %%
# Latent variable for the car centric attitude
car_explanatory_variables: dict[str, Expression] = {
    'age_65_more': age_65_more,
    'ScaledIncome': ScaledIncome,
    'moreThanOneCar': moreThanOneCar,
    'moreThanOneBike': moreThanOneBike,
    'individualHouse': individualHouse,
    'haveChildren': haveChildren,
    'haveGA': haveGA,
    'highEducation': highEducation,
}
# %%
# Latent variable for the urban preference attitude
urban_explanatory_variables: dict[str, Expression] = {
    'childCenter': childCenter,
    'childSuburb': childSuburb,
    'highEducation': highEducation,
    'artisans': SocioProfCat == 5,
    'employees': SocioProfCat == 6,
    'age_30_less': age <= 30,
    'haveChildren': haveChildren,
    'UrbRur': UrbRur,
    'IndividualHouse': HouseType == 1,
}

# %%
# Dict of all explanatory variables
all_explanatory_variables = car_explanatory_variables | urban_explanatory_variables
