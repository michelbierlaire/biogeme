"""

Latent variables
================

Definitions of latent variables used in the hybrid choice model.

This module centralizes the specification of latent variables that enter the
hybrid choice (MIMIC) model. For each latent variable, it defines:

- the name of the latent variable,
- the list of explanatory variables entering its structural equation, and
- the set of Likert-type indicators used in its measurement equations.

The goal is to keep all latent-variable metadata in a single, transparent
location, making the model specification easier to read, maintain, and
modify.

The variables defined here are imported by higher-level model construction
code and should therefore remain lightweight and declarative (no model logic
is implemented in this file).

Michel Bierlaire
Thu Dec 25 2025, 08:13:19
"""

"""Latent variable representing the car-centric attitude.

This latent variable captures preferences and attitudes related to car
ownership and car-oriented lifestyles. It is explained by socio-demographic
and background variables and measured using a set of mobility, lifestyle,
and environment-related Likert indicators.
"""
car_explanatory_variables: list[str] = [
    'high_education',
    'top_manager',
    'employees',
    'age_30_less',
    'ScaledIncome',
    'car_oriented_parents',
]

car_name = 'car_centric_attitude'
car_likert_indicators: set[str] = {
    'Envir01',
    'Envir02',
    'Envir06',
    'Mobil03',
    'Mobil05',
    'Mobil08',
    'Mobil09',
    'Mobil10',
    'LifSty07',
    'NbCar',
}

"""Latent variable representing the environmental attitude.

This latent variable captures environmental awareness and sensitivity. Its
structural equation depends on socio-demographic and residential background
variables, and it is measured using a set of environment-, mobility-, and
lifestyle-related Likert indicators.
"""
environment_explanatory_variables: list[str] = [
    'childSuburb',
    'ScaledIncome',
    'city_center_as_kid',
    'artisans',
    'high_education',
    'low_education',
]

env_name = 'environmental_attitude'
environment_likert_indicators: set[str] = {
    'Envir01',
    'Envir02',
    'Envir03',
    'Envir04',
    'Envir05',
    'Envir06',
    'Mobil12',
    'LifSty01',
}
