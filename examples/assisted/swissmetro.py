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
    Elem,
    Numeric,
)

from biogeme.assisted import (
    DiscreteSegmentationTuple,
    TermTuple,
    SegmentedParameterTuple,
)


## Step 1: data preparation. Identical to any Biogeme script.
logger = msg.bioMessage()
logger.setDetailed()

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

CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_COST = database.DefineVariable('TRAIN_COST', TRAIN_CO * (GA == 0) / 100)
SM_COST = database.DefineVariable('SM_COST', SM_CO * (GA == 0) / 100)
CAR_COST = database.DefineVariable('CAR_COST', CAR_CO / 100)


## Step 2: identify and name the relevant attributes of the alternatives
attributes = {
    'Train travel time': TRAIN_TT_SCALED,
    'Swissmetro travel time': SM_TT_SCALED,
    'Car travel time': CAR_TT_SCALED,
    'Train travel cost': TRAIN_COST,
    'Swissmetro travel cost': SM_COST,
    'Car travel cost': CAR_COST,
    'Train headway': TRAIN_HE,
    'Swissmetro headway': SM_HE,
}

## Step 3: define the group of attributes

# Group the attributes. All attributes in the same group will be
# associated with the same transformation, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise
groupsOfAttributes = {
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

## Step 4: force some groups of attributes to be alternative specific.

# In this example, all the attributes can potentially be generic
genericForbiden = None

## Step 5: force some groups of attributes to be active.

# In this example, we impose time and cost to be in the model
forceActive = ['Travel time', 'Travel cost']


## Step 6: define potential transformations of the attributes


def mylog(x):
    """Log of the attribute, or 0 if it is zero

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return 'log', Elem({0: log(x), 1: Numeric(0)}, x == 0)


def sqrt(x):
    """Sqrt of the attribute

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)

    """
    return 'sqrt', x**0.5


def square(x):
    """Square of the attribute

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)

    """
    return 'square', x**2


def piecewise(x, thresholds, name):
    """Piecewise linear transformation of a variable

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :param thresholds: list of thresholds
    :type thresholds: list(float)

    :param name: name of the variable
    :type name: str
    """
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
    return (f'piecewise_{name}_{thresholds}', formula)


def piecewise_time_1(x):
    """Piecewise linear for time :math:`0, 0.1, +\\infty`

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)

    """
    return piecewise(x, [0, 0.1, None], 'time')


def piecewise_time_2(x):
    """Piecewise linear for time :math:`0, 0.25, +\\infty`

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return piecewise(x, [0, 0.25, None], 'time')


def piecewise_cost_1(x):
    """Piecewise linear for cost :math:`0, 0.1, +\\infty`
    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)

    """
    return piecewise(x, [0, 0.1, None], 'cost')


def piecewise_cost_2(x):
    """Piecewise linear for cost :math:`0, 0.25, +\\infty`

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return piecewise(x, [0, 0.25, None], 'cost')


def boxcox(x, name):
    """Box-Cox transform of the attribute. This is not a valid
        transformation, as it has two arguments. It is defined to be
        called by another function.

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :param name: name of the variable to be transformed
    :type name: str

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    ell = Beta(f'lambda_{name}', 1, None, None, 0)
    return f'Box-Cox_{name}', models.boxcox(x, ell)


def boxcox_time(x):
    """Box-Cox transform of the attribute time. This is a valid transformation.
    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return boxcox(x, 'time')


def boxcox_cost(x):
    """Box-Cox transform of the attribute cost

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return boxcox(x, 'cost')


def boxcox_headway(x):
    """Box-Cox transform of the attribute headway

    :param x: attribute
    :type x: biogeme.expressions.Variable

    :return: name of the transformation, and expression to calculate it.
    :rtype: tuple(str, biogeme.expressions.Expression)
    """
    return boxcox(x, 'headway')


# Associate each group of attributes with possible
# transformations. Define a dictionary where the keys are the names of
# the groups of attributes, and the values are lists of functions
# defined in the previous step.

transformations = {
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

## Step 7: define the potential segmentations

segmentations_cte = {
    'GA': DiscreteSegmentationTuple(variable=GA, mapping={1: 'GA', 0: 'noGA'}),
    'gender': DiscreteSegmentationTuple(
        variable=MALE, mapping={0: 'female', 1: 'male'}
    ),
    'class': DiscreteSegmentationTuple(
        variable=FIRST, mapping={0: 'secondClass', 1: 'firstClass'}
    ),
    'luggage': DiscreteSegmentationTuple(
        variable=LUGGAGE, mapping={0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}
    ),
    'income': DiscreteSegmentationTuple(
        variable=INCOME,
        mapping={
            1: 'inc-under50',
            2: 'inc-50-100',
            3: 'inc-100+',
            4: 'inc-unknown',
        },
    ),
}

segmentations_cost = {
    'GA': DiscreteSegmentationTuple(variable=GA, mapping={1: 'GA', 0: 'noGA'}),
    'gender': DiscreteSegmentationTuple(
        variable=MALE, mapping={0: 'female', 1: 'male'}
    ),
    'income': DiscreteSegmentationTuple(
        variable=INCOME,
        mapping={
            1: 'inc-under50',
            2: 'inc-50-100',
            3: 'inc-100+',
            4: 'inc-unknown',
        },
    ),
    'class': DiscreteSegmentationTuple(
        variable=FIRST, mapping={0: 'secondClass', 1: 'firstClass'}
    ),
    'who': DiscreteSegmentationTuple(
        variable=WHO,
        mapping={1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'},
    ),
}

segmentations_time = {
    'GA': DiscreteSegmentationTuple(variable=GA, mapping={1: 'GA', 0: 'noGA'}),
    'gender': DiscreteSegmentationTuple(
        variable=MALE, mapping={0: 'female', 1: 'male'}
    ),
    'class': DiscreteSegmentationTuple(
        variable=FIRST, mapping={0: 'secondClass', 1: 'firstClass'}
    ),
    'luggage': DiscreteSegmentationTuple(
        variable=LUGGAGE, mapping={0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}
    ),
    'who': DiscreteSegmentationTuple(
        variable=WHO,
        mapping={1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'},
    ),
}

segmentations_headway = {
    'class': DiscreteSegmentationTuple(
        variable=FIRST, mapping={0: 'secondClass', 1: 'firstClass'}
    ),
    'luggage': DiscreteSegmentationTuple(
        variable=LUGGAGE, mapping={0: 'noLugg', 1: 'oneLugg', 3: 'severalLugg'}
    ),
    'who': DiscreteSegmentationTuple(
        variable=WHO,
        mapping={1: 'egoPays', 2: 'employerPays', 3: 'fiftyFifty'},
    ),
}


segmentations = {
    'Seg. cte': SegmentedParameterTuple(
        dict=segmentations_cte, combinatorial=False
    ),
    'Seg. cost': SegmentedParameterTuple(
        dict=segmentations_cost, combinatorial=False
    ),
    'Seg. time': SegmentedParameterTuple(
        dict=segmentations_time, combinatorial=False
    ),
    'Seg. headway': SegmentedParameterTuple(
        dict=segmentations_headway, combinatorial=False
    ),
}


## Step 8: Specification of the utility function. For each term, it is possible
## to define bounds on the coefficient, and to include a function that
## verifies its validity a posteriori.

utility_train = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='Train travel time',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Train travel cost',
        segmentation='Seg. cost',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Train headway',
        segmentation='Seg. headway',
        bounds=(None, 0),
        validity=None,
    ),
]

utility_sm = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='Swissmetro travel time',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Swissmetro travel cost',
        segmentation='Seg. cost',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Swissmetro headway',
        segmentation='Seg. headway',
        bounds=(None, 0),
        validity=None,
    ),
]

utility_car = [
    TermTuple(
        attribute='Car travel time',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Car travel cost',
        segmentation='Seg. cost',
        bounds=(None, 0),
        validity=None,
    ),
]

utilities = {
    1: ('train', utility_train),
    2: ('Swissmetro', utility_sm),
    3: ('car', utility_car),
}

## Step 9: availabilities
availabilities = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Step 10: We define potential candidates for the choice model.
def logit(V, av, choice):
    """logit model

    :param V: the dictionary of utility functions.
    :type V: dict(int, biogeme.expressions.Expression)

    :param av: the dictionary of availability conditions
    :type av: dict(int, biogeme.expressions.Expression)

    :param choice: the expression to calculate the chosen alternative
    :type choice: biogeme.expressions.Expression

    :return: expression representing the contribution of each
        observation to the log likelihood function
    :rtype: biogeme.expressions.Expression

    """
    return models.loglogit(V, av, choice)


def nested1(V, av, choice):
    """Nested logit model: existing / future
    :param V: the dictionary of utility functions.
    :type V: dict(int, biogeme.expressions.Expression)

    :param av: the dictionary of availability conditions
    :type av: dict(int, biogeme.expressions.Expression)

    :param choice: the expression to calculate the chosen alternative
    :type choice: biogeme.expressions.Expression

    :return: expression representing the contribution of each
        observation to the log likelihood function
    :rtype: biogeme.expressions.Expression

    """
    existing = Beta('mu_existing', 1, 1, None, 0), [1, 3]
    future = 1.0, [2]
    nests = existing, future
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    """Nested logit model: public / private

    :param V: the dictionary of utility functions.
    :type V: dict(int, biogeme.expressions.Expression)

    :param av: the dictionary of availability conditions
    :type av: dict(int, biogeme.expressions.Expression)

    :param choice: the expression to calculate the chosen alternative
    :type choice: biogeme.expressions.Expression

    :return: expression representing the contribution of each
        observation to the log likelihood function
    :rtype: biogeme.expressions.Expression

    """
    public = Beta('mu_public', 1, 1, None, 0), [1, 2]
    private = 1.0, [3]
    nests = public, private
    return models.lognested(V, av, nests, choice)


def cnl1(V, av, choice):
    """Cross nested logit: fixed alphas

    :param V: the dictionary of utility functions.
    :type V: dict(int, biogeme.expressions.Expression)

    :param av: the dictionary of availability conditions
    :type av: dict(int, biogeme.expressions.Expression)

    :param choice: the expression to calculate the chosen alternative
    :type choice: biogeme.expressions.Expression

    :return: expression representing the contribution of each
        observation to the log likelihood function
    :rtype: biogeme.expressions.Expression
    """
    mu_existing = Beta('mu_existing', 1, 1, None, 0)
    mu_public = Beta('mu_public', 1, 1, None, 0)
    alpha_existing = {1: 0.5, 2: 0, 3: 1}
    alpha_public = {1: 0.5, 2: 1, 3: 0}
    nest_existing = mu_existing, alpha_existing
    nest_public = mu_public, alpha_public
    nests = nest_existing, nest_public
    return models.logcnl_avail(V, av, nests, choice)


def cnl2(V, av, choice):
    """Cross nested logit: fixed alphas

    :param V: the dictionary of utility functions.
    :type V: dict(int, biogeme.expressions.Expression)

    :param av: the dictionary of availability conditions
    :type av: dict(int, biogeme.expressions.Expression)

    :param choice: the expression to calculate the chosen alternative
    :type choice: biogeme.expressions.Expression

    :return: expression representing the contribution of each
        observation to the log likelihood function
    :rtype: biogeme.expressions.Expression
    """
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

## Step 11:  Definition of the specification problem, gathering all information
# defined above.
theProblem = assisted.specificationProblem(
    'Swissmetro',
    database,
    attributes,
    groupsOfAttributes,
    genericForbiden,
    forceActive,
    transformations,
    segmentations,
    utilities,
    availabilities,
    CHOICE,
    myModels,
)

theProblem.maximumNumberOfParameters = 300

# We propose several specifications to initialize the algorithm.
# First model: three alternative specific attributes
attr = {
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
    theProblem.generateSolution(attr, sg, 'Logit'),
    theProblem.generateSolution(attr, sg, 'Nested one stop'),
    theProblem.generateSolution(attr, sg, 'Nested same'),
    theProblem.generateSolution(attr, sg, 'CNL alpha fixed'),
    theProblem.generateSolution(attr, sg, 'CNL alpha est.'),
]


# Optimization algorithm
vns.vns(
    theProblem,
    initSolutions,
    archiveInputFile='swissmetroPareto.pickle',
    pickleOutputFile='swissmetroPareto.pickle',
)
