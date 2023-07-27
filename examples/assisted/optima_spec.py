"""File optima_spec.py

:author: Michel Bierlaire, EPFL
:date: Sat Jul 22 16:58:54 2023

Specification for the assisted specification algorithm. Optima case study.
"""
from biogeme import models
import biogeme.biogeme as bio
from biogeme.expressions import (
    Beta,
    logzero,
)
from biogeme.catalog import (
    Catalog,
    segmentation_catalogs,
    generic_alt_specific_catalogs,
)

from optima_data import (
    database,
    TimePT_scaled,
    TimeCar_scaled,
    MarginalCostPT_scaled,
    CostCarCHF_scaled,
    distance_km_scaled,
    WaitingTimePT,
    TripPurpose,
    UrbRur,
    LangCode,
    Gender,
    OccupStat,
    subscription,
    CarAvail,
    Education,
    NbTransf,
    Choice,
    age,
)


# Nonlinear specifications

ell_time = Beta('lambda_ttime', 0, -10, 10, 0)

pt_travel_time_catalog = Catalog.from_dict(
    catalog_name='pt_ttime',
    dict_of_expressions={
        'linear': TimePT_scaled,
        'log': logzero(TimePT_scaled),
        'sqrt': TimePT_scaled**0.5,
        'square': TimePT_scaled**2,
        'boxcox': models.boxcox(TimePT_scaled, ell_time),
        'piecewise': models.piecewise_as_variable(
            TimePT_scaled, [0, 30 / 200, 60 / 200, None]
        ),
    },
)

car_travel_time_catalog = Catalog.from_dict(
    catalog_name='car_ttime',
    dict_of_expressions={
        'linear': TimeCar_scaled,
        'log': logzero(TimeCar_scaled),
        'sqrt': TimeCar_scaled**0.5,
        'square': TimeCar_scaled**2,
        'boxcox': models.boxcox(TimeCar_scaled, ell_time),
        'piecewise': models.piecewise_as_variable(
            TimeCar_scaled, [0, 30 / 200, 60 / 200, None]
        ),
    },
    controlled_by=pt_travel_time_catalog.controlled_by,
)

ell_wait = Beta('lambda_waiting', 0, -10, 10, 0)

waiting_time_catalog = Catalog.from_dict(
    catalog_name='waiting',
    dict_of_expressions={
        'zero': 0,
        'linear': WaitingTimePT,
        'log': logzero(WaitingTimePT),
        'sqrt': WaitingTimePT**0.5,
        'square': WaitingTimePT**2,
        'boxcox': models.boxcox(WaitingTimePT, ell_wait),
    },
)

ell_age = Beta('lambda_age', 0, -10, 10, 0)
waiting_time_age_catalog = Catalog.from_dict(
    catalog_name='waiting_age',
    dict_of_expressions={
        'no_age': waiting_time_catalog,
        'age': waiting_time_catalog * (age / 100) ** ell_age,
    },
)

# Segmentations

# Define all possible segmentations
all_discrete_segmentations = {
    'TripPurpose': database.generate_segmentation(
        variable=TripPurpose, mapping={1: 'work', 2: 'others'}
    ),
    'Urban': database.generate_segmentation(
        variable=UrbRur, mapping={1: 'rural', 2: 'urban'}
    ),
    'Language': database.generate_segmentation(
        variable=LangCode, mapping={1: 'French', 2: 'German'}
    ),
    'Gender': database.generate_segmentation(
        variable=Gender, mapping={1: 'male', 2: 'female', -1: 'unkown'}
    ),
    'Occupation': database.generate_segmentation(
        variable=OccupStat,
        mapping={1: 'full_time', 2: 'partial_time', 3: 'others'},
    ),
    'Subscription': database.generate_segmentation(
        variable=subscription, mapping={0: 'none', 1: 'GA', 2: 'other'}
    ),
    'CarAvail': database.generate_segmentation(
        variable=CarAvail, mapping={1: 'yes', 3: 'no'}
    ),
    'Education': database.generate_segmentation(
        variable=Education,
        mapping={
            3: 'vocational',
            4: 'high_school',
            6: 'higher_edu',
            7: 'university',
        },
    ),
}

tuple_of_segmentations = tuple(all_discrete_segmentations.values())

cte_pt = Beta('cte_pt', 0, None, None, 0)
cte_car = Beta('cte_car', 0, None, None, 0)
cte_pt_catalog, cte_car_catalog = segmentation_catalogs(
    generic_name='cte',
    beta_parameters=[cte_pt, cte_car],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)


beta_time = Beta('b_tt', 0, None, 0, 0)
(beta_time_catalog,) = generic_alt_specific_catalogs(
    generic_name='beta_time',
    beta_parameters=[beta_time],
    alternatives=['pt', 'car'],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)


# We normalize beta_cost to -1 for the moneymetric specification
beta_cost = Beta('b_cost', -1, None, 0, 1)

beta_wait = Beta('b_wait', 0, None, 0, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_wait_catalog,) = segmentation_catalogs(
    generic_name='beta_wait',
    beta_parameters=[beta_wait],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_dist = Beta('b_dist', 0, None, 0, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_dist_catalog,) = segmentation_catalogs(
    generic_name='beta_dist',
    beta_parameters=[beta_dist],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)


beta_transfer = Beta('b_transf', 0, None, 0, 0)


mu = Beta('mu_ga', 1, 0.00001, None, 0)
(mu_catalog,) = segmentation_catalogs(
    generic_name='mu',
    beta_parameters=[mu],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

V_pt = (
    cte_pt_catalog
    + beta_time_catalog['pt'] * pt_travel_time_catalog
    + beta_cost * MarginalCostPT_scaled
    + beta_transfer * NbTransf
    + beta_wait_catalog * waiting_time_catalog
)

V_car = (
    cte_car_catalog
    + beta_time_catalog['car'] * car_travel_time_catalog
    + beta_cost * CostCarCHF_scaled
)

V_sm = beta_dist_catalog * distance_km_scaled


V = {0: mu_catalog * V_pt, 1: mu_catalog * V_car, 2: mu_catalog * V_sm}
av = {0: 1, 1: CarAvail != 3, 2: 1}

# Nests
private = Beta('mu_existing', 1, 1, 10, 0), [1, 2]
public = 1.0, [0]
nests_pp = private, public

fast = Beta('mu_existing', 1, 1, 10, 0), [0, 1]
slow = 1.0, [2]
nests_fs = fast, slow

model_catalog = Catalog.from_dict(
    catalog_name='model',
    dict_of_expressions={
        'logit': models.loglogit(V, av, Choice),
        'nested_pp': models.lognested(V, av, nests_pp, Choice),
        'nested_fs': models.lognested(V, av, nests_fs, Choice),
    },
)

the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'optima'
