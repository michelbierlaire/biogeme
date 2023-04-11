"""File optima_spec.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 11:29:18 2023

Specification for the assisted specification algorithm. Optima case study.
"""
from biogeme import models
import biogeme.biogeme as bio
from biogeme.expressions import (
    Beta,
    log,
    logzero,
)
from biogeme.catalog import Catalog, SynchronizedCatalog, segmentation_catalog
from biogeme.segmentation import DiscreteSegmentationTuple

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
)


# Nonlinear specifications

ell_time = Beta('lambda_ttime', 0, None, None, 0)

pt_travel_time_catalog = Catalog.from_dict(
    catalog_name='pt_ttime',
    dict_of_expressions={
        'linear': TimePT_scaled,
        'dist_interaction': TimePT_scaled * log(1 + distance_km_scaled / 1000),
        'log': logzero(TimePT_scaled),
        'sqrt': TimePT_scaled**0.5,
        'square': TimePT_scaled**2,
        'boxcox': models.boxcox(TimePT_scaled, ell_time),
    },
)

car_travel_time_catalog = SynchronizedCatalog.from_dict(
    catalog_name='car_ttime',
    dict_of_expressions={
        'linear': TimeCar_scaled,
        'dist_interaction': TimeCar_scaled * log(1 + distance_km_scaled / 1000),
        'log': logzero(TimeCar_scaled),
        'sqrt': TimeCar_scaled**0.5,
        'square': TimeCar_scaled**2,
        'boxcox': models.boxcox(TimeCar_scaled, ell_time),
    },
    controller=pt_travel_time_catalog,
)

ell_cost = Beta('lambda_cost', 0, None, None, 0)

pt_cost_catalog = Catalog.from_dict(
    catalog_name='pt_cost',
    dict_of_expressions={
        'linear': MarginalCostPT_scaled,
        'log': logzero(MarginalCostPT_scaled),
        'sqrt': MarginalCostPT_scaled**0.5,
        'square': MarginalCostPT_scaled**2,
        'boxcox': models.boxcox(MarginalCostPT_scaled, ell_cost),
    },
)

car_cost_catalog = SynchronizedCatalog.from_dict(
    catalog_name='pt_cost',
    dict_of_expressions={
        'linear': CostCarCHF_scaled,
        'log': logzero(CostCarCHF_scaled),
        'sqrt': CostCarCHF_scaled**0.5,
        'square': CostCarCHF_scaled**2,
        'boxcox': models.boxcox(CostCarCHF_scaled, ell_cost),
    },
    controller=pt_cost_catalog,
)

ell_wait = Beta('lambda_waiting', 0, None, None, 0)

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

# Segmentations

# Define all possible segmentations
all_discrete_segmentations = {
    'TripPurpose': DiscreteSegmentationTuple(
        variable=TripPurpose, mapping={1: 'work', 2: 'others'}
    ),
    'Urban': DiscreteSegmentationTuple(
        variable=UrbRur, mapping={1: 'rural', 2: 'urban'}
    ),
    'Language': DiscreteSegmentationTuple(
        variable=LangCode, mapping={1: 'French', 2: 'German'}
    ),
    'Gender': DiscreteSegmentationTuple(
        variable=Gender, mapping={1: 'male', 2: 'female', -1: 'unkown'}
    ),
    'Occupation': DiscreteSegmentationTuple(
        variable=OccupStat,
        mapping={1: 'full_time', 2: 'partial_time', 3: 'others'},
    ),
    'Subscription': DiscreteSegmentationTuple(
        variable=subscription, mapping={0: 'none', 1: 'GA', 2: 'other'}
    ),
    'CarAvail': DiscreteSegmentationTuple(
        variable=CarAvail, mapping={1: 'yes', 3: 'no'}
    ),
    'Education': DiscreteSegmentationTuple(
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
cte_pt_catalog = segmentation_catalog(
    beta_parameter=cte_pt,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

cte_car = Beta('cte_pt', 0, None, None, 0)
cte_car_catalog = segmentation_catalog(
    beta_parameter=cte_car,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
    synchronized_with=cte_pt_catalog,
)

beta_time = Beta('b_tt', 0, None, 0, 0)
beta_time_catalog = segmentation_catalog(
    beta_parameter=beta_time,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_time_pt = Beta('b_tt_pt', 0, None, 0, 0)
beta_time_pt_catalog = segmentation_catalog(
    beta_parameter=beta_time_pt,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_time_car = Beta('b_tt_car', 0, None, 0, 0)
beta_time_car_catalog = segmentation_catalog(
    beta_parameter=beta_time_car,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
    synchronized_with=beta_time_pt_catalog,
)

# Genric or alternative specific

beta_time_pt_generic = Catalog.from_dict(
    catalog_name='b_time_pt_generic',
    dict_of_expressions={
        'generic': beta_time_catalog,
        'altspec': beta_time_pt_catalog,
    },
)

beta_time_car_generic = SynchronizedCatalog.from_dict(
    catalog_name='b_time_car_generic',
    dict_of_expressions={
        'generic': beta_time_catalog,
        'altspec': beta_time_car_catalog,
    },
    controller=beta_time_pt_generic,
)

beta_cost = Beta('b_cost', 0, None, 0, 0)
beta_cost_catalog = segmentation_catalog(
    beta_parameter=beta_cost,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_wait = Beta('b_wait', 0, None, 0, 0)
beta_wait_catalog = segmentation_catalog(
    beta_parameter=beta_wait,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_dist = Beta('b_dist', 0, None, 0, 0)
beta_dist_catalog = segmentation_catalog(
    beta_parameter=beta_dist,
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)


beta_transfer = Beta('b_transf', 0, None, 0, 0)
V_pt = (
    cte_pt_catalog
    + beta_time_pt_generic * pt_travel_time_catalog
    + beta_cost_catalog * pt_cost_catalog
    + beta_transfer * NbTransf
    + beta_wait_catalog * waiting_time_catalog
)

V_car = (
    cte_car_catalog
    + beta_time_car_generic * car_travel_time_catalog
    + beta_cost_catalog * car_cost_catalog
)

V_sm = beta_dist_catalog * distance_km_scaled

V = {0: V_pt, 1: V_car, 2: V_sm}
av = {0: 1, 1: CarAvail != 3, 2: 1}

# Nests
private = Beta('mu_existing', 1, 1, None, 0), [1, 2]
public = 1.0, [0]
nests_pp = private, public

fast = Beta('mu_existing', 1, 1, None, 0), [0, 1]
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
