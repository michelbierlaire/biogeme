"""File specification.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 10:57:50 2024

Specification of the baseline utilities of a MDCEV model.
"""

from biogeme.expressions import Beta
from process_data import (
    database,
    PersonID,
    weight as orig_weight,
    hhsize,
    childnum,
    faminc,
    faminc25K,
    income,
    employed,
    fulltime,
    spousepr,
    spousemp,
    male,
    married,
    age,
    age2,
    age15_40,
    age41_60,
    age61_85,
    bachigher,
    white,
    metro,
    diaryday,
    Sunday,
    holiday,
    weekearn,
    weekwordur,
    hhchild,
    ohhchild,
    t1,
    t2,
    t3,
    t4,
    number_chosen,
)

# %
# Estimation will be carried out with WESML. The weights are defined here.
weight = orig_weight * 1.7718243289995812

# %
# Parameters to be estimated
cte_shopping = Beta('cte_shopping', 0, None, None, 0)
cte_socializing = Beta('cte_socializing', 0, None, None, 0)
cte_recreation = Beta('cte_recreation', 0, None, None, 0)

number_members_socializing = Beta('number_members_socializing', 0, None, None, 0)
number_members_recreation = Beta('number_members_recreation', 0, None, None, 0)

metropolitan_shopping = Beta('metropolitan_shopping', 0, None, None, 0)

male_shopping = Beta('male_shopping', 0, None, None, 0)
male_socializing = Beta('male_socializing', 0, None, None, 0)
male_recreation = Beta('male_recreation', 0, None, None, 0)

age_15_40_shopping = Beta('age_15_40_shopping', 0, None, None, 0)
age_15_40_recreation = Beta('age_15_40_recreation', 0, None, None, 0)

age_41_60_socializing = Beta('age_41_60_socializing', 0, None, None, 0)
age_41_60_personal = Beta('age_41_60_personal', 0, None, None, 0)

bachelor_socializing = Beta('bachelor_socializing', 0, None, None, 0)
bachelor_personal = Beta('bachelor_personal', 0, None, None, 0)

white_personal = Beta('white_personal', 0, None, None, 0)

spouse_shopping = Beta('spouse_shopping', 0, None, None, 0)
spouse_recreation = Beta('spouse_recreation', 0, None, None, 0)

employed_shopping = Beta('employed_shopping', 0, None, None, 0)

sunday_socializing = Beta('sunday_socializing', 0, None, None, 0)
sunday_personal = Beta('sunday_personal', 0, None, None, 0)

# %
# Definition of the utility functions
shopping = (
    cte_shopping
    + metropolitan_shopping * metro
    + male_shopping * male
    + age_15_40_shopping * age15_40
    + spouse_shopping * spousepr
    + employed_shopping * employed
)

socializing = (
    cte_socializing
    + number_members_socializing * hhsize
    + male_socializing * male
    + age_41_60_socializing * age41_60
    + bachelor_socializing * bachigher
    + sunday_socializing * Sunday
)

recreation = (
    cte_recreation
    + number_members_recreation * hhsize
    + male_recreation * male
    + age_15_40_recreation * age15_40
    + spouse_recreation * spousepr
)

personal = (
    age_41_60_personal * age41_60
    + bachelor_personal * bachigher
    + white_personal * white
    + sunday_personal * Sunday
)

# %
# Group the utility functions into a dictionary, mapping the alternative id with the specification.
baseline_utilities = {
    1: shopping,
    2: socializing,
    3: recreation,
    4: personal,
}

# %
# Observed consumed quantities.
consumed_quantities = {
    1: t1 / 60.0,
    2: t2 / 60.0,
    3: t3 / 60.0,
    4: t4 / 60.0,
}
