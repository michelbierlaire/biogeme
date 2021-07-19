"""File swissmetro.py

:author: Michel Bierlaire, EPFL
:date: Mon Dec 21 16:04:28 2020

Assisted specification for the Swissmetro case study
"""

# Too constraining
# pylint: disable=invalid-name

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
)

logger = msg.bioMessage()
logger.setDebug()

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')

# Update some data
df['TRAIN_TT_SCALED'] = df['TRAIN_TT'] / 100
df['TRAIN_TT_SQUARE'] = df['TRAIN_TT_SCALED'] * df['TRAIN_TT_SCALED'] / 100
df['SM_TT_SCALED'] = df['SM_TT'] / 100
df['SM_TT_SQUARE'] = df['SM_TT_SCALED'] * df['SM_TT_SCALED'] / 100
df['CAR_TT_SCALED'] = df['CAR_TT'] / 100
df['CAR_TT_SQUARE'] = df['CAR_TT_SCALED'] * df['CAR_TT_SCALED'] / 100


database = db.Database('swissmetro', df)
globals().update(database.variables)

exclude = (CHOICE == 0) > 0
database.remove(exclude)

# Definition of new variables

CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0), database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0), database)
TRAIN_COST = DefineVariable('TRAIN_COST', TRAIN_CO * (GA == 0) / 100, database)
SM_COST = DefineVariable('SM_COST', SM_CO * (GA == 0) / 100, database)
CAR_COST = DefineVariable('CAR_COST', CAR_CO / 100, database)


# Definition of potential nonlinear transforms of variables
def mylog(x):
    """Log of the variable, or 0 if the variable is zero"""
    return 'log', Elem({0: log(x), 1: Numeric(0)}, x == 0)


def sqrt(x):
    """Sqrt of the variable"""
    return 'sqrt', x ** 0.5


def square(x):
    """Square of the variable"""
    return 'square', x ** 2


def piecewise(x, thresholds, name):
    """Piecewise linear specification"""
    piecewiseVariables = models.piecewiseVariables(x, thresholds)
    formula = piecewiseVariables[0]
    for k in range(1, len(thresholds) - 1):
        formula += (
            Beta(
                f'pw_{name}_{thresholds[k-1]}_{thresholds[k]}',
                0,
                None,
                None,
                0,
            )
            * piecewiseVariables[k]
        )
    return (f'piecewise_{thresholds}', formula)


def piecewise_time_1(x):
    """Piecewise linear for time :math:`0, 0.1, +\\infty`"""
    return piecewise(x, [0, 0.1, None], 'time')


def piecewise_time_2(x):
    """Piecewise linear for time :math:`0, 0.25, +\\infty`"""
    return piecewise(x, [0, 0.25, None], 'time')


def piecewise_cost_1(x):
    """Piecewise linear for cost :math:`0, 0.1, +\\infty`"""
    return piecewise(x, [0, 0.1, None], 'cost')


def piecewise_cost_2(x):
    """Piecewise linear for cost :math:`0, 0.25, +\\infty`"""
    return piecewise(x, [0, 0.25, None], 'cost')


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


def boxcox_headway(x):
    """Box-Cox transform of the variable headway"""
    return boxcox(x, 'headway')


# Define all possible segmentations
segmentations_cte = {
    'GA': (GA, {1: 'GA', 0: 'noGA'}),
    'gender': (MALE, {0: 'female', 1: 'male'}),
    'class': (FIRST, {0: 'secondClass', 1: 'firstClass'}),
    'luggage': (LUGGAGE, {0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}),
    'income': (
        INCOME,
        {1: 'inc-under50', 2: 'inc-50-100', 3: 'inc-100+', 4: 'inc-unknown'},
    ),
}

segmentations_cost = {
    'GA': (GA, {1: 'GA', 0: 'noGA'}),
    'gender': (MALE, {0: 'female', 1: 'male'}),
    'income': (
        INCOME,
        {1: 'inc-under50', 2: 'inc-50-100', 3: 'inc-100+', 4: 'inc-unknown'},
    ),
    'class': (FIRST, {0: 'secondClass', 1: 'firstClass'}),
    'who': (WHO, {1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'}),
}

segmentations_time = {
    'GA': (GA, {1: 'GA', 0: 'noGA'}),
    'gender': (MALE, {0: 'female', 1: 'male'}),
    'who': (WHO, {1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'}),
    'class': (FIRST, {0: 'secondClass', 1: 'firstClass'}),
    'luggage': (LUGGAGE, {0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}),
}

segmentations_headway = {
    'class': (FIRST, {0: 'secondClass', 1: 'firstClass'}),
    'luggage': (LUGGAGE, {0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}),
    'who': (WHO, {1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'}),
}


segmentations = {
    'Seg. cte': segmentations_cte,
    'Seg. cost': segmentations_cost,
    'Seg. time': segmentations_time,
    'Seg. headway': segmentations_headway,
}


# Define the attributes of the alternatives
variables = {
    'Train travel time': TRAIN_TT_SCALED,
    'Swissmetro travel time': SM_TT_SCALED,
    'Car travel time': CAR_TT_SCALED,
    'Train travel cost': TRAIN_COST,
    'Swissmetro travel cost': SM_COST,
    'Car travel cost': CAR_COST,
    'Train headway': TRAIN_HE,
    'Swissmetro headway': SM_HE,
}


# Group the attributes. All attributes in the same group will be
# assoaited with the same nonlinear transform, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise
groupsOfVariables = {
    'Travel time': [
        'Train travel time',
        'Swissmetro travel time',
        'Car travel time',
    ],
    'Travel cost': [
        'Train travel cost',
        'Swissmetro travel cost',
        'Car travel cost',
    ],
    'Headway': ['Train headway', 'Swissmetro headway'],
}


# In this example, all the variables could be generic
genericForbiden = None

# In this example, we impose time and cost ot be in the model
forceActive = ['Travel time', 'Travel cost']

# Associate a list of potential nonlinearities with each group of variable
nonlinearSpecs = {
    'Travel time': [
        mylog,
        sqrt,
        square,
        piecewise_time_1,
        piecewise_time_2,
        boxcox_time,
    ],
    'Travel cost': [
        mylog,
        sqrt,
        square,
        piecewise_cost_1,
        piecewise_cost_2,
        boxcox_cost,
    ],
    'Headway': [mylog, sqrt, square, boxcox_headway],
}

# Specification of the utility function. For each term, it is possible
# to define bounds on the coefficient, and to include a function that
# verifies its validity a posteriori.

utility_train = [
    (None, 'Seg. cte', (None, None), None),
    ('Train travel time', 'Seg. time', (None, 0), None),
    ('Train travel cost', 'Seg. cost', (None, 0), None),
    ('Train headway', 'Seg. headway', (None, 0), None),
]

utility_sm = [
    (None, 'Seg. cte', (None, None), None),
    ('Swissmetro travel time', 'Seg. time', (None, 0), None),
    ('Swissmetro travel cost', 'Seg. cost', (None, 0), None),
    ('Swissmetro headway', 'Seg. headway', (None, 0), None),
]

utility_car = [
    ('Car travel time', 'Seg. time', (None, 0), None),
    ('Car travel cost', 'Seg. cost', (None, 0), None),
]


utilities = {
    1: ('train', utility_train),
    2: ('Swissmetro', utility_sm),
    3: ('car', utility_car),
}

availabilities = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# We define potential candidates for the choice model.
def logit(V, av, choice):
    """logit model"""
    return models.loglogit(V, av, choice)


def nested1(V, av, choice):
    """Nested logit model: existing / future"""
    existing = Beta('mu_existing', 1, 1, None, 0), [1, 3]
    future = 1.0, [2]
    nests = existing, future
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    """Nested logit model: public / private"""
    public = Beta('mu_public', 1, 1, None, 0), [1, 2]
    private = 1.0, [3]
    nests = public, private
    return models.lognested(V, av, nests, choice)


def cnl1(V, av, choice):
    """Cross nested logit: fixed alphas"""
    mu_existing = Beta('mu_existing', 1, 1, None, 0)
    mu_public = Beta('mu_public', 1, 1, None, 0)
    alpha_existing = {1: 0.5, 2: 0, 3: 1}
    alpha_public = {1: 0.5, 2: 1, 3: 0}
    nest_existing = mu_existing, alpha_existing
    nest_public = mu_public, alpha_public
    nests = nest_existing, nest_public
    return models.logcnl_avail(V, av, nests, choice)


def cnl2(V, av, choice):
    """Cross nested logit: fixed alphas"""
    alpha = Beta('alpha', 0.5, 0, 1, 0)
    mu_existing = Beta('mu_existing', 1, 1, None, 0)
    mu_public = Beta('mu_public', 1, 1, None, 0)
    alpha_existing = {1: alpha, 2: 0, 3: 1}
    alpha_public = {1: 1 - alpha, 2: 1, 3: 0}
    nest_existing = mu_existing, alpha_existing
    nest_public = mu_public, alpha_public
    nests = nest_existing, nest_public
    return models.logcnl_avail(V, av, nests, choice)


# We provide names to these candidates
myModels = {
    'Logit': logit,
    'Nested one stop': nested1,
    'Nested same': nested2,
    'CNL alpha fixed': cnl1,
    'CNL alpha est.': cnl2,
}

# Definition of the specification problem, gathering all information
# defined above.
theProblem = assisted.specificationProblem(
    'Swissmetro',
    database,
    variables,
    groupsOfVariables,
    genericForbiden,
    forceActive,
    nonlinearSpecs,
    segmentations,
    utilities,
    availabilities,
    CHOICE,
    myModels,
)

theProblem.maximumNumberOfParameters = 300

# First model: three alternative specific attributes
nl = {
    'Travel time': (0, False),
    'Travel cost': (0, False),
    'Headway': (0, False),
}

sg = {
    'Seg. cte': ['GA'],
    'Seg. cost': ['class', 'who'],
    'Seg. time': ['gender'],
    'Seg. headway': ['class'],
}

initSolutions = [
    theProblem.generateSolution(nl, sg, 'Logit'),
    theProblem.generateSolution(nl, sg, 'Nested one stop'),
    theProblem.generateSolution(nl, sg, 'Nested same'),
    theProblem.generateSolution(nl, sg, 'CNL alpha fixed'),
    theProblem.generateSolution(nl, sg, 'CNL alpha est.'),
]


# Optimization algorithm
vns.vns(
    theProblem,
    initSolutions,
    archiveInputFile='airlinePareto.pickle',
    pickleOutputFile='airlinePareto.pickle',
)
