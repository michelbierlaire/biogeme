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
    DefineVariable,
    Elem,
    Numeric,
    Variable,
)

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


otherSubscription = DefineVariable(
    'otherSubscription',
    ((HalfFareST == 1) + (LineRelST == 1) + (AreaRelST == 1) + (OtherST) == 1)
    > 0,
    database,
)

subscription = DefineVariable(
    'subscription',
    (GenAbST == 1) * 1 + (GenAbST != 1) * otherSubscription * 2,
    database,
)

TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5

# Definition of potential nonlinear transforms of variables
def mylog(x):
    """Log of the variable"""
    return 'log', Elem({0: log(x), 1: Numeric(0)}, x == 0)


def sqrt(x):
    """Sqrt of the variable"""
    return 'sqrt', x ** 0.5


def square(x):
    """Square of the variable"""
    return 'square', x ** 2


def boxcox(x, name):
    """Box-Cox transform of the variable"""
    ell = Beta(f'lambda_{name}', 1, 0.0001, 3.0, 0)
    return f'Box-Cox_{name}', models.boxcox(x, ell)


def boxcox_time(x):
    """Box-Cox transform of the variable time"""
    return boxcox(x, 'time')


def boxcox_cost(x):
    """Box-Cox transform of the variable cost"""
    return boxcox(x, 'cost')


def distanceInteraction(x):
    """Nonlinea rinteraction with distance"""
    return 'dist. interaction', x * log(1 + Variable('distance_km') / 1000)


# Define all possible segmentations
all_segmentations = {
    'TripPurpose': (TripPurpose, {1: 'work', 2: 'others'}),
    'Urban': (UrbRur, {1: 'rural', 2: 'urban'}),
    'Language': (LangCode, {1: 'French', 2: 'German'}),
    'Income': (
        Income,
        {
            1: '<2500',
            2: '2051_4000',
            3: '4001_6000',
            4: '6001_8000',
            5: '8001_10000',
            6: '>10000',
            -1: 'unknown',
        },
    ),
    'Gender': (Gender, {1: 'male', 2: 'female', -1: 'unkown'}),
    'Occupation': (
        OccupStat,
        {1: 'full_time', 2: 'partial_time', 3: 'others'},
    ),
    'Subscription': (subscription, {0: 'none', 1: 'GA', 2: 'other'}),
    'CarAvail': (CarAvail, {1: 'yes', 3: 'no'}),
    'Education': (
        Education,
        {3: 'vocational', 4: 'high_school', 6: 'higher_edu', 7: 'university'},
    ),
}


# Define segmentations
segmentations = {
    'Seg. cte': all_segmentations,
    'Seg. cost': all_segmentations,
    'Seg. wait': all_segmentations,
    'Seg. time': all_segmentations,
    'Seg. transfers': all_segmentations,
    'Seg. dist': all_segmentations,
}

# Define the attributes of the alternatives
variables = {
    'PT travel time': TimePT_scaled,
    'PT travel cost': MarginalCostPT_scaled,
    'Car travel time': TimeCar_scaled,
    'Car travel cost': CostCarCHF_scaled,
    'Distance': distance_km_scaled,
    'Transfers': NbTransf,
    'PT Waiting time': WaitingTimePT,
}

# Group the attributes. All attributes in the same group will be
# assoaited with the same nonlinear transform, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise
groupsOfVariables = {
    'Travel time': ['PT travel time', 'Car travel time'],
    'Travel cost': ['PT travel cost', 'Car travel cost'],
}

# In this example, no variable must be alternative specific
genericForbiden = None

# In this example, all the variables must be in the model
forceActive = ['Travel time', 'Distance']

# Associate a list of potential nonlinearities with each group of variable
nonlinearSpecs = {
    'Travel time': [distanceInteraction, mylog, sqrt, square, boxcox_time],
    'PT Waiting time': [mylog, sqrt, square, boxcox_time],
    'Travel cost': [mylog, sqrt, square, boxcox_cost],
}

def negativeParameter(val):
    """ Function verifying the negativity of the parameters"""
    return val < 0


# Specification of the utility function. For each term, it is possible
# to define bounds on the coefficient, and to include a function that
# verifies its validity a posteriori.

utility_pt = [
    (None, 'Seg. cte', (None, None), None),
    ('PT travel time', 'Seg. time', (None, 0), negativeParameter),
    ('PT travel cost', 'Seg. cost', (None, 0), negativeParameter),
    ('Transfers', 'Seg. transfers', (None, 0), negativeParameter),
    ('PT Waiting time', 'Seg. wait', (None, 0), negativeParameter),
]


utility_car = [
    (None, 'Seg. cte', (None, None), None),
    ('Car travel time', 'Seg. time', (None, 0), negativeParameter),
    ('Car travel cost', 'Seg. cost', (None, 0), negativeParameter),
]

utility_sm = [('Distance', 'Seg. dist', (None, 0), negativeParameter)]


utilities = {
    0: ('pt', utility_pt),
    1: ('car', utility_car),
    2: ('sm', utility_sm),
}

availabilities = {0: 1, 1: CarAvail != 3, 2: 1}


# We define potential candidates for the choice model.
def logit(V, av, choice):
    """logit model"""
    return models.loglogit(V, av, choice)


def nested1(V, av, choice):
    """Nested logit model: first specification """
    same = Beta('mu_same', 1, 1, None, 0), [0, 1]
    multiple = 1.0, [2]
    nests = same, multiple
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    """Nested logit model: second specification """
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
myModels = {'Logit': logit,
            'Nested 1': nested1,
            'Nested 2': nested2,
            'Cross nested 1': cnl1,
            'Cross nested 2': cnl2}

# Definition of the specification problem, gathering all information
# defined above.
theProblem = assisted.specificationProblem(
    'Optima',
    database,
    variables,
    groupsOfVariables,
    genericForbiden,
    forceActive,
    nonlinearSpecs,
    segmentations,
    utilities,
    availabilities,
    Choice,
    myModels,
)

theProblem.maximumNumberOfParameters = 100

# We propose several specifications to initialize the algorithm.
# For each group of attributes, we decide if it is nonlinear, and generic.
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
