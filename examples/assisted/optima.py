"""File optima.py

:author: Michel Bierlaire, EPFL
:date: Thu Dec 24 16:51:28 2020

Assisted specification for the Optima case study
"""

# Too constraining
# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
from biogeme import models
import biogeme.messaging as msg
from biogeme import vns
from biogeme import assisted
from biogeme.expressions import (
    Beta,
    log,
    Elem,
    Numeric,
    Variable,
)
from biogeme.assisted import (
    DiscreteSegmentationTuple,
    TermTuple,
    SegmentedParameterTuple,
)


## Step 1: data preparation. Identical to any Biogeme script.
logger = msg.bioMessage()
logger.setDebug()

# Read the data
df = pd.read_csv('optima.dat', sep='\t')

df.loc[df['OccupStat'] > 2, 'OccupStat'] = 3
df.loc[df['OccupStat'] == -1, 'OccupStat'] = 3

df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] == 5, 'Education'] = 4
df.loc[df['Education'] == 8, 'Education'] = 7

df.loc[df['TripPurpose'] != 1, 'TripPurpose'] = 2

df.loc[df['CarAvail'] != 3, 'CarAvail'] = 1

database = db.Database('optima', df)

globals().update(database.variables)

exclude = (
    (Choice == -1) + (CostCarCHF < 0) + (CarAvail == 3) * (Choice == 1)
) > 0
database.remove(exclude)

# Definition of new variables


otherSubscription = database.DefineVariable(
    'otherSubscription',
    ((HalfFareST == 1) + (LineRelST == 1) + (AreaRelST == 1) + (OtherST) == 1)
    > 0,
)

subscription = database.DefineVariable(
    'subscription', (GenAbST == 1) * 1 + (GenAbST != 1) * otherSubscription * 2
)

TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5

## Step 2: attributes

# Define the attributes of the alternatives
attributes = {
    'PT travel time': TimePT_scaled,
    'PT travel cost': MarginalCostPT_scaled,
    'Car travel time': TimeCar_scaled,
    'Car travel cost': CostCarCHF_scaled,
    'Distance': distance_km_scaled,
    'Transfers': NbTransf,
    'PT Waiting time': WaitingTimePT,
}

## Step 3:Group the attributes. All attributes in the same group will be
# associated with the same  transform, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise
groupsOfAttributes = {
    'Travel time': ['PT travel time', 'Car travel time'],
    'Travel cost': ['PT travel cost', 'Car travel cost'],
}

## Step 4
# In this example, no group of attributes must be alternative specific
genericForbiden = None

## Step 5
# In this example, all the attributes must be in the model
forceActive = ['Travel time', 'Distance']


## Step 6: Definition of potential transforms of attributes
def mylog(x):
    """Log of the attribute"""
    return 'log', Elem({0: log(x), 1: Numeric(0)}, x == 0)


def sqrt(x):
    """Sqrt of the attribute"""
    return 'sqrt', x**0.5


def square(x):
    """Square of the attribute"""
    return 'square', x**2


def boxcox(x, name):
    """Box-Cox transform of the attribute"""
    ell = Beta(f'lambda_{name}', 1, 0.0001, 3.0, 0)
    return f'Box-Cox_{name}', models.boxcox(x, ell)


def boxcox_time(x):
    """Box-Cox transform of the attribute time"""
    return boxcox(x, 'time')


def boxcox_cost(x):
    """Box-Cox transform of the attribute cost"""
    return boxcox(x, 'cost')


def distanceInteraction(x):
    """Nonlinear interaction with distance"""
    return ('dist. interaction', x * log(1 + Variable('distance_km') / 1000))


# Associate a list of potential transformations with each group of attributes
transformations = {
    'Travel time': [distanceInteraction, mylog, sqrt, square, boxcox_time],
    'PT Waiting time': [mylog, sqrt, square, boxcox_time],
    'Travel cost': [mylog, sqrt, square, boxcox_cost],
}

## Step 7

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

# Continuous segmentations

# Define segmentations
segmentations = {
    'Seg. cte': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
    'Seg. cost': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
    'Seg. wait': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
    'Seg. time': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
    'Seg. transfers': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
    'Seg. dist': SegmentedParameterTuple(
        dict=all_discrete_segmentations, combinatorial=False
    ),
}


## Step 8: utility function

# First, we define a function that checks if a parameter is negative.
def negativeParameter(val):
    """Function verifying the negativity of the coefficient.

    :param val: value to verify
    :type val: float

    :return: True if the value is negative, False otherwise.
    :rtype: bool
    """
    return val < 0


# Specification of the utility function. For each term, it is possible
# to define bounds on the coefficient, and to include a function that
# verifies its validity a posteriori.

utility_pt = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='PT travel time',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
    TermTuple(
        attribute='PT travel cost',
        segmentation='Seg. cost',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
    TermTuple(
        attribute='Transfers',
        segmentation='Seg. transfers',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
    TermTuple(
        attribute='PT Waiting time',
        segmentation='Seg. wait',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
]


utility_car = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='Car travel time',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
    TermTuple(
        attribute='Car travel cost',
        segmentation='Seg. cost',
        bounds=(None, 0),
        validity=negativeParameter,
    ),
]

utility_sm = [
    TermTuple(
        attribute='Distance',
        segmentation='Seg. dist',
        bounds=(None, 0),
        validity=negativeParameter,
    )
]


utilities = {
    0: ('pt', utility_pt),
    1: ('car', utility_car),
    2: ('sm', utility_sm),
}

## Step 9
availabilities = {0: 1, 1: CarAvail != 3, 2: 1}

## Step 10
# We define potential candidates for the choice model.
def logit(V, av, choice):
    """logit model"""
    return models.loglogit(V, av, choice)


def nested1(V, av, choice):
    """Nested logit model: first specification"""
    same = Beta('mu_same', 1, 1, None, 0), [0, 1]
    multiple = 1.0, [2]
    nests = same, multiple
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    """Nested logit model: second specification"""
    onestop = Beta('mu_onestop', 1, 1, None, 0), [1, 2]
    nostop = 1.0, [0]
    nests = nostop, onestop
    return models.lognested(V, av, nests, choice)


def cnl1(V, av, choice):
    """Cross nested logit: fixed alphas"""
    mu_same = Beta('mu_same', 1, 1, None, 0)
    mu_onestop = Beta('mu_onestop', 1, 1, None, 0)
    alpha_same = {0: 1.0, 1: 0.5, 2: 0}
    alpha_onestop = {0: 0, 1: 0.5, 2: 1}
    nest_same = mu_same, alpha_same
    nest_onestop = mu_onestop, alpha_onestop
    nests = nest_onestop, nest_same
    return models.logcnl_avail(V, av, nests, choice)


def cnl2(V, av, choice):
    """Cross nested logit: estimated alphas"""
    alpha = Beta('alpha', 0.5, 0, 1, 0)
    mu_same = Beta('mu_same', 1, 1, None, 0)
    mu_onestop = Beta('mu_onestop', 1, 1, None, 0)
    alpha_same = {0: 1.0, 1: alpha, 2: 0}
    alpha_onestop = {0: 0, 1: 1 - alpha, 2: 1}
    nest_same = mu_same, alpha_same
    nest_onestop = mu_onestop, alpha_onestop
    nests = nest_onestop, nest_same
    return models.logcnl_avail(V, av, nests, choice)


# We provide names to these candidates
myModels = {
    'Logit': logit,
    'Nested 1': nested1,
    'Nested 2': nested2,
    'Cross nested 1': cnl1,
    'Cross nested 2': cnl2,
}

## Step 11
# Definition of the specification problem, gathering all information
# defined above.
theProblem = assisted.specificationProblem(
    'Optima',
    database,
    attributes,
    groupsOfAttributes,
    genericForbiden,
    forceActive,
    transformations,
    segmentations,
    utilities,
    availabilities,
    Choice,
    myModels,
)

theProblem.maximumNumberOfParameters = 100

# We propose several specifications to initialize the algorithm.
# For each group of attributes, we decide if it is transformed, and generic.
nl1 = {'Travel time': (None, False), 'Distance': (None, False)}

# For each segmentation, we decided which dimensions are active.
sg1 = {
    'Seg. cte': ['Subscription'],
    'Seg. time': [],
    'Seg. cost': [],
    'Seg. dist': [],
}

initSolutions = [theProblem.generateSolution(nl1, sg1, 'Logit')]

# Optimization algorithm
vns.vns(
    theProblem,
    initSolutions,
    archiveInputFile='optimaPareto.pickle',
    pickleOutputFile='optimaPareto.pickle',
)
