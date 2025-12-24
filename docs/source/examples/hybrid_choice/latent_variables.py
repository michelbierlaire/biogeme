# Latent variable for the car centric attitude
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

# Latent variable for the environmental attitude
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
    'NbCar',
}
