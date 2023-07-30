"""File airline_spec.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 16:22:32 2023

Specification for the assisted specification algorithm. Airline case study
"""


from biogeme import models
import biogeme.biogeme as bio
from biogeme.catalog import (
    Catalog,
    segmentation_catalogs,
    generic_alt_specific_catalogs,
)
from biogeme.expressions import (
    Beta,
    log,
    logzero,
    Variable,
)

from airline_data import (
    database,
    chosenAlternative,
    Fare_1,
    Fare_2,
    Fare_3,
    Legroom_1,
    Legroom_2,
    Legroom_3,
    TripTimeHours_1,
    TripTimeHours_2,
    TripTimeHours_3,
    Opt1_SchedDelayEarly,
    Opt2_SchedDelayEarly,
    Opt3_SchedDelayEarly,
    Opt1_SchedDelayLate,
    Opt2_SchedDelayLate,
    Opt3_SchedDelayLate,
    q02_TripPurpose,
    q03_WhoPays,
    q17_Gender,
    q11_DepartureOrArrivalIsImportant,
    education,
)

# Define the catalog for the time variable

ell_fare = Beta('lambda_fare', 1, None, 10, 0)

fare_direct_catalog = Catalog.from_dict(
    catalog_name='fare_direct_catalog',
    dict_of_expressions={
        'linear': Fare_1,
        'income_interaction_1': Fare_1 / Variable('Cont_Income'),
        'income_interaction_2': Fare_1 + Fare_1 / Variable('Cont_Income'),
        'log_income_interaction': log(Fare_1) / Variable('Cont_Income'),
        'sqrt_income_interaction': Fare_1**0.5 / Variable('Cont_Income'),
        'log': logzero(Fare_1),
        'sqrt': Fare_1**0.5,
        'square': Fare_1 * Fare_1,
        'boxcox': models.boxcox(Fare_1, ell_fare),
    },
)

fare_same_catalog = Catalog.from_dict(
    catalog_name='fare_same_catalog',
    dict_of_expressions={
        'linear': Fare_2,
        'income_interaction_1': Fare_2 / Variable('Cont_Income'),
        'income_interaction_2': Fare_2 + Fare_2 / Variable('Cont_Income'),
        'log_income_interaction': log(Fare_2) / Variable('Cont_Income'),
        'sqrt_income_interaction': Fare_2**0.5 / Variable('Cont_Income'),
        'log': logzero(Fare_2),
        'sqrt': Fare_2**0.5,
        'square': Fare_2 * Fare_2,
        'boxcox': models.boxcox(Fare_2, ell_fare),
    },
    controlled_by=fare_direct_catalog.controlled_by,
)

fare_multiple_catalog = Catalog.from_dict(
    catalog_name='fare_multiple_catalog',
    dict_of_expressions={
        'linear': Fare_3,
        'income_interaction_1': Fare_3 / Variable('Cont_Income'),
        'income_interaction_2': Fare_3 + Fare_3 / Variable('Cont_Income'),
        'log_income_interaction': log(Fare_3) / Variable('Cont_Income'),
        'sqrt_income_interaction': Fare_3**0.5 / Variable('Cont_Income'),
        'log': logzero(Fare_3),
        'sqrt': Fare_3**0.5,
        'square': Fare_3 * Fare_3,
        'boxcox': models.boxcox(Fare_3, ell_fare),
    },
    controlled_by=fare_direct_catalog.controlled_by,
)

legroom_direct_catalog = Catalog.from_dict(
    catalog_name='legroom_direct_catalog',
    dict_of_expressions={
        'linear': Legroom_1,
        'log': logzero(Legroom_1),
        'sqrt': Legroom_1**0.5,
        'square': Legroom_1 * Legroom_1,
    },
)

legroom_same_catalog = Catalog.from_dict(
    catalog_name='legroom_same_catalog',
    dict_of_expressions={
        'linear': Legroom_2,
        'log': logzero(Legroom_2),
        'sqrt': Legroom_2**0.5,
        'square': Legroom_2 * Legroom_2,
    },
    controlled_by=legroom_direct_catalog.controlled_by,
)

legroom_multiple_catalog = Catalog.from_dict(
    catalog_name='legroom_multiple_catalog',
    dict_of_expressions={
        'linear': Legroom_3,
        'log': logzero(Legroom_3),
        'sqrt': Legroom_3**0.5,
        'square': Legroom_3 * Legroom_3,
    },
    controlled_by=legroom_direct_catalog.controlled_by,
)

ell_time = Beta('lambda_time', 1, None, 10, 0)

time_direct_catalog = Catalog.from_dict(
    catalog_name='time_direct_catalog',
    dict_of_expressions={
        'linear': TripTimeHours_1,
        'log': logzero(TripTimeHours_1),
        'sqrt': TripTimeHours_1**0.5,
        'square': TripTimeHours_1 * TripTimeHours_1,
        'boxcox': models.boxcox(TripTimeHours_1, ell_time),
        'piecewise_1': models.piecewise_as_variable(
            TripTimeHours_1, [0, 2, 4, 6, 8, None]
        ),
        'piecewise_2': models.piecewise_as_variable(TripTimeHours_1, [0, 2, 8, None]),
    },
)

time_same_catalog = Catalog.from_dict(
    catalog_name='time_same_catalog',
    dict_of_expressions={
        'linear': TripTimeHours_2,
        'log': logzero(TripTimeHours_2),
        'sqrt': TripTimeHours_2**0.5,
        'square': TripTimeHours_2 * TripTimeHours_2,
        'boxcox': models.boxcox(TripTimeHours_2, ell_time),
        'piecewise_1': models.piecewise_as_variable(
            TripTimeHours_2, [0, 2, 4, 6, 8, None]
        ),
        'piecewise_2': models.piecewise_as_variable(TripTimeHours_2, [0, 2, 8, None]),
    },
    controlled_by=time_direct_catalog.controlled_by,
)

time_multiple_catalog = Catalog.from_dict(
    catalog_name='time_multiple_catalog',
    dict_of_expressions={
        'linear': TripTimeHours_3,
        'log': logzero(TripTimeHours_3),
        'sqrt': TripTimeHours_3**0.5,
        'square': TripTimeHours_3 * TripTimeHours_3,
        'boxcox': models.boxcox(TripTimeHours_3, ell_time),
        'piecewise_1': models.piecewise_as_variable(
            TripTimeHours_3, [0, 2, 4, 6, 8, None]
        ),
        'piecewise_2': models.piecewise_as_variable(TripTimeHours_3, [0, 2, 8, None]),
    },
    controlled_by=time_direct_catalog.controlled_by,
)

early_direct_catalog = Catalog.from_dict(
    catalog_name='early_direct_catalog',
    dict_of_expressions={
        'linear': Opt1_SchedDelayEarly,
        'log': logzero(Opt1_SchedDelayEarly),
        'sqrt': Opt1_SchedDelayEarly**0.5,
        'square': Opt1_SchedDelayEarly * Opt1_SchedDelayEarly,
    },
)

early_same_catalog = Catalog.from_dict(
    catalog_name='early_same_catalog',
    dict_of_expressions={
        'linear': Opt2_SchedDelayEarly,
        'log': logzero(Opt2_SchedDelayEarly),
        'sqrt': Opt2_SchedDelayEarly**0.5,
        'square': Opt2_SchedDelayEarly * Opt2_SchedDelayEarly,
    },
    controlled_by=early_direct_catalog.controlled_by,
)

early_multiple_catalog = Catalog.from_dict(
    catalog_name='early_multiple_catalog',
    dict_of_expressions={
        'linear': Opt3_SchedDelayEarly,
        'log': logzero(Opt3_SchedDelayEarly),
        'sqrt': Opt3_SchedDelayEarly**0.5,
        'square': Opt3_SchedDelayEarly * Opt3_SchedDelayEarly,
    },
    controlled_by=early_direct_catalog.controlled_by,
)

late_direct_catalog = Catalog.from_dict(
    catalog_name='late_direct_catalog',
    dict_of_expressions={
        'linear': Opt1_SchedDelayLate,
        'log': logzero(Opt1_SchedDelayLate),
        'sqrt': Opt1_SchedDelayLate**0.5,
        'square': Opt1_SchedDelayLate * Opt1_SchedDelayLate,
    },
    controlled_by=early_direct_catalog.controlled_by,
)

late_same_catalog = Catalog.from_dict(
    catalog_name='late_same_catalog',
    dict_of_expressions={
        'linear': Opt2_SchedDelayLate,
        'log': logzero(Opt2_SchedDelayLate),
        'sqrt': Opt2_SchedDelayLate**0.5,
        'square': Opt2_SchedDelayLate * Opt2_SchedDelayLate,
    },
    controlled_by=early_direct_catalog.controlled_by,
)

late_multiple_catalog = Catalog.from_dict(
    catalog_name='late_multiple_catalog',
    dict_of_expressions={
        'linear': Opt3_SchedDelayLate,
        'log': logzero(Opt3_SchedDelayLate),
        'sqrt': Opt3_SchedDelayLate**0.5,
        'square': Opt3_SchedDelayLate * Opt3_SchedDelayLate,
    },
    controlled_by=early_direct_catalog.controlled_by,
)

# Define the potential segmentations
all_segmentations = {
    'TripPurpose': database.generate_segmentation(
        variable=q02_TripPurpose,
        mapping={
            1: 'business',
            2: 'leisure',
            3: 'attending conf.',
            4: 'business & leisure',
            0: 'unknown',
        },
    ),
    'Gender': database.generate_segmentation(
        variable=q17_Gender,
        mapping={1: 'male', 2: 'female', -1: 'unknown'},
    ),
    'Education': database.generate_segmentation(
        variable=education,
        mapping={
            1: 'high_edu',
            2: 'medium_edu',
            3: 'low_edu',
            -1: 'unknown',
        },
    ),
    'Importance': database.generate_segmentation(
        variable=q11_DepartureOrArrivalIsImportant,
        mapping={1: 'departure', 2: 'arrival', 0: 'not important'},
    ),
    'Who pays': database.generate_segmentation(
        variable=q03_WhoPays,
        mapping={1: 'traveler', 2: 'employer', 3: 'third party', 0: 'unknown'},
    ),
}

tuple_of_segmentations = tuple(all_segmentations.values())


cte_same = Beta('cte_same', 0, None, None, 0)
cte_multiple = Beta('cte_multiple', 0, None, None, 0)
cte_same_catalog, cte_multiple_catalog = segmentation_catalogs(
    generic_name='cte',
    beta_parameters=[cte_same, cte_multiple],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_fare = Beta('beta_fare', 0, None, 0, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_fare_catalog,) = segmentation_catalogs(
    generic_name='beta_fare',
    beta_parameters=[beta_fare],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_time = Beta('beta_time', 0, None, 0, 0)

(beta_time_catalog,) = generic_alt_specific_catalogs(
    generic_name='beta_time',
    beta_parameters=[beta_time],
    alternatives=['direct', 'same', 'multiple'],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_early = Beta('beta_early', 0, None, 0, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_early_catalog,) = segmentation_catalogs(
    generic_name='beta_early',
    beta_parameters=[beta_early],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_late = Beta('beta_late', 0, None, 0, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_late_catalog,) = segmentation_catalogs(
    generic_name='beta_late',
    beta_parameters=[beta_late],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)

beta_legroom = Beta('beta_legroom', 0, 0, None, 0)
# The function returns a list. In this case, the list has only one
# element. The variable is followed by a comma so that the only
# catalog in the list is directly extracted and assigned to the
# variable.
(beta_legroom_catalog,) = segmentation_catalogs(
    generic_name='beta_legroom',
    beta_parameters=[beta_legroom],
    potential_segmentations=tuple_of_segmentations,
    maximum_number=3,
)


utility_direct = (
    beta_fare_catalog * fare_direct_catalog
    + beta_legroom_catalog * legroom_direct_catalog
    + beta_early_catalog * early_direct_catalog
    + beta_late_catalog * late_direct_catalog
    + beta_time_catalog['direct'] * time_direct_catalog
)

utility_same = (
    cte_same_catalog
    + beta_fare_catalog * fare_same_catalog
    + beta_legroom_catalog * legroom_same_catalog
    + beta_early_catalog * early_same_catalog
    + beta_late_catalog * late_same_catalog
    + beta_time_catalog['same'] * time_same_catalog
)

utility_multiple = (
    cte_multiple_catalog
    + beta_fare_catalog * fare_multiple_catalog
    + beta_legroom_catalog * legroom_multiple_catalog
    + beta_early_catalog * early_multiple_catalog
    + beta_late_catalog * late_multiple_catalog
    + beta_time_catalog['multiple'] * time_multiple_catalog
)

V = {
    1: utility_direct,
    2: utility_same,
    3: utility_multiple,
}

# Step 9: availabilities
av = {1: 1, 2: 1, 3: 1}

# Nests
onestop = Beta('mu_onestop', 1, 1, 10, 0), [2, 3]
nonstop = 1.0, [1]
nests_1 = nonstop, onestop

same = Beta('mu_same', 1, 1, 10, 0), [1, 2]
multiple = 1.0, [3]
nests_2 = same, multiple

mu_onestop = Beta('mu_onestop', 1, 1, 10, 0)
mu_same = Beta('mu_same', 1, 1, 10, 0)
alpha_onestop = {1: 1.0, 2: 0.5, 3: 1}
alpha_same = {1: 1.0, 2: 0.5, 3: 1}
nest_onestop = mu_onestop, alpha_onestop
nest_same = mu_same, alpha_same
cnl_nests_1 = nest_onestop, nest_same

alpha = Beta('alpha', 0.5, 0, 1, 0)
mu_onestop = Beta('mu_onestop', 1, 1, 10, 0)
mu_same = Beta('mu_same', 1, 1, 10, 0)
alpha_onestop = {1: 1.0, 2: alpha, 3: 1}
alpha_same = {1: 1.0, 2: 1 - alpha, 3: 1}
nest_onestop = mu_onestop, alpha_onestop
nest_same = mu_same, alpha_same
cnl_nests_2 = nest_onestop, nest_same


model_catalog = Catalog.from_dict(
    catalog_name='model',
    dict_of_expressions={
        'logit': models.loglogit(V, av, chosenAlternative),
        'nested_onestop': models.lognested(V, av, nests_1, chosenAlternative),
        'nested_same': models.lognested(V, av, nests_2, chosenAlternative),
        'CNL_alpha_fixed': models.logcnl_avail(V, av, cnl_nests_1, chosenAlternative),
        'CNL_alpha_est': models.logcnl_avail(V, av, cnl_nests_2, chosenAlternative),
    },
)

the_biogeme = bio.BIOGEME(database, model_catalog)
the_biogeme.modelName = 'airline'
