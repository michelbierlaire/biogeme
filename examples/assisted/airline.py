"""File airline.py

:author: Michel Bierlaire, EPFL
:date: Mon Dec 21 15:24:50 2020

Assisted specification for the airline cases tudy
"""

# Too constraining
# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.messaging as msg
from biogeme import models, vns, assisted
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
df = pd.read_csv('airline.dat', sep='\t')

# Update some data
df.loc[df['q17_Gender'] == 99, 'q17_Gender'] = -1
df.loc[df['q20_Education'] == 99, 'q20_Education'] = -1

database = db.Database('airline', df)
globals().update(database.variables)

exclude = ArrivalTimeHours_1 == -1
database.remove(exclude)

# Definition of new variables

chosenAlternative = (
    (BestAlternative_1 * 1) + (BestAlternative_2 * 2) + (BestAlternative_3 * 3)
)

DepartureTimeSensitive = database.DefineVariable(
    'DepartureTimeSensitive', q11_DepartureOrArrivalIsImportant == 1
)
ArrivalTimeSensitive = database.DefineVariable(
    'ArrivalTimeSensitive', q11_DepartureOrArrivalIsImportant == 2
)

DesiredDepartureTime = database.DefineVariable(
    'DesiredDepartureTime', q12_IdealDepTime
)
DesiredArrivalTime = database.DefineVariable(
    'DesiredArrivalTime', q13_IdealArrTime
)
ScheduledDelay_1 = database.DefineVariable(
    'ScheduledDelay_1',
    (DepartureTimeSensitive * (DepartureTimeMins_1 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_1 - DesiredArrivalTime)),
)

ScheduledDelay_2 = database.DefineVariable(
    'ScheduledDelay_2',
    (DepartureTimeSensitive * (DepartureTimeMins_2 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_2 - DesiredArrivalTime)),
)

ScheduledDelay_3 = database.DefineVariable(
    'ScheduledDelay_3',
    (DepartureTimeSensitive * (DepartureTimeMins_3 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_3 - DesiredArrivalTime)),
)

Opt1_SchedDelayEarly = database.DefineVariable(
    'Opt1_SchedDelayEarly', (-(ScheduledDelay_1) * (ScheduledDelay_1 < 0)) / 60
)
Opt2_SchedDelayEarly = database.DefineVariable(
    'Opt2_SchedDelayEarly', (-(ScheduledDelay_2) * (ScheduledDelay_2 < 0)) / 60
)
Opt3_SchedDelayEarly = database.DefineVariable(
    'Opt3_SchedDelayEarly', (-(ScheduledDelay_3) * (ScheduledDelay_3 < 0)) / 60
)

Opt1_SchedDelayLate = database.DefineVariable(
    'Opt1_SchedDelayLate', (ScheduledDelay_1 * (ScheduledDelay_1 > 0)) / 60
)
Opt2_SchedDelayLate = database.DefineVariable(
    'Opt2_SchedDelayLate', (ScheduledDelay_2 * (ScheduledDelay_2 > 0)) / 60
)
Opt3_SchedDelayLate = database.DefineVariable(
    'Opt3_SchedDelayLate', (ScheduledDelay_3 * (ScheduledDelay_3 > 0)) / 60
)

## Step 2: identify and name the relevant attributes of the alternatives
# Define the attributes of the alternatives

attributes = {
    'Fare direct': Fare_1,
    'Fare same': Fare_2,
    'Fare multiple': Fare_3,
    'Legroom direct': Legroom_1,
    'Legroom same': Legroom_2,
    'Legroom multiple': Legroom_3,
    'Time direct': TripTimeHours_1,
    'Time same': TripTimeHours_2,
    'Time multiple': TripTimeHours_3,
    'Early direct': Opt1_SchedDelayEarly,
    'Early same': Opt2_SchedDelayEarly,
    'Early multiple': Opt3_SchedDelayEarly,
    'Late direct': Opt1_SchedDelayLate,
    'Late same': Opt2_SchedDelayLate,
    'Late multiple': Opt3_SchedDelayLate,
}

## Step 3: define the group of attributes

# Group the attributes. All attributes in the same group will be
# associated with the same transformation, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise

groupsOfAttributes = {
    'Fare': ['Fare direct', 'Fare same', 'Fare multiple'],
    'Legroom': ['Legroom direct', 'Legroom same', 'Legroom multiple'],
    'Time': ['Time direct', 'Time same', 'Time multiple'],
    'Early': ['Early direct', 'Early same', 'Early multiple'],
    'Late': ['Late direct', 'Late same', 'Late multiple'],
}

## Step 4: force some groups of attributes to be alternative specific.
# In this example, no variable must be alternative specific
genericForbiden = None

## Step 5: force some groups of attributes to be active.
# In this example, all the variables must be in the model
forceActive = list(groupsOfAttributes.keys())

## Step 6: define potential transformations of the attributes
def incomeInteraction(x):
    """Defines an interaction with income"""
    return 'inc. interaction', x / Variable('Cont_Income')


def incomeInteraction2(x):
    """Defines another interaction with income"""
    return 'inc. interaction 2', x + x / Variable('Cont_Income')


def logincomeInteraction(x):
    """Defines an interaction with between the log and income"""
    return 'inc. interaction', log(x) / Variable('Cont_Income')


def sqrtincomeInteraction(x):
    """Defines an interaction with between the sqrt and income"""
    return 'inc. interaction', x**0.5 / Variable('Cont_Income')


def mylog(x):
    """Log of the variable"""
    return 'log', Elem({0: log(x), 1: Numeric(0)}, x == 0)


def sqrt(x):
    """Sqrt of the variable"""
    return 'sqrt', x**0.5


def square(x):
    """Square of the variable"""
    return 'square', x**2


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


def piecewise_time_2(x):
    """Piecewise linear for time :math:`0, 2, 8, +\\infty`"""
    return piecewise(x, [0, 2, 8, None], 'time')


def piecewise_time_1(x):
    """Piecewise linear for time :math:`0, 2, 4, 6, 8, +\\infty`"""
    return piecewise(x, [0, 2, 4, 6, 8, None], 'time')


def boxcox(x, name):
    """Box-Cox transform of the variable"""
    ell = Beta(f'lambda_{name}', 1, 0.0001, 3.0, 0)
    return f'Box-Cox_{name}', models.boxcox(x, ell)


def boxcox_time(x):
    """Box-Cox transform of the variable time"""
    return boxcox(x, 'time')


def boxcox_fare(x):
    """Box-Cox transform of the variable fare"""
    return boxcox(x, 'fare')


## Step 7: Associate each group of attributes with possible
## transformations. Define a dictionary where the keys are the names
## of the groups of attributes, and the values are lists of functions
## defined in the previous step.
nonlinearSpecs = {
    'Time': [
        mylog,
        sqrt,
        square,
        boxcox_time,
        piecewise_time_1,
        piecewise_time_2,
    ],
    'Fare': [
        incomeInteraction,
        incomeInteraction2,
        logincomeInteraction,
        sqrtincomeInteraction,
        mylog,
        sqrt,
        square,
        boxcox_fare,
    ],
    'Legroom': [mylog, sqrt, square],
    'Early': [mylog, sqrt, square],
    'Late': [mylog, sqrt, square],
}


## Step 7: define the potential segmentations
all_segmentations = {
    'TripPurpose': DiscreteSegmentationTuple(
        variable=q02_TripPurpose,
        mapping={
            1: 'business',
            2: 'leisure',
            3: 'attending conf.',
            4: 'business & leisure',
            0: 'unknown',
        },
    ),
    'Gender': DiscreteSegmentationTuple(
        variable=q17_Gender,
        mapping={1: 'male', 2: 'female', -1: 'unknown'},
    ),
    'Education': DiscreteSegmentationTuple(
        variable=q20_Education,
        mapping={
            1: 'less than high school',
            2: 'high school',
            3: 'some college',
            4: 'associate occ.',
            5: 'associate acad.',
            6: 'bachelor',
            7: 'master',
            8: 'professional',
            9: 'doctorate',
            -1: 'unkonown',
        },
    ),
    'Importance': DiscreteSegmentationTuple(
        variable=q11_DepartureOrArrivalIsImportant,
        mapping={1: 'departure', 2: 'arrival', 3: 'not important'},
    ),
    'Who pays': DiscreteSegmentationTuple(
        variable=q03_WhoPays,
        mapping={1: 'traveler', 2: 'employer', 3: 'third party', 0: 'unknown'},
    ),
}

# Define segmentations
segmentations = {
    'Seg. cte': SegmentedParameterTuple(
        dict=all_segmentations, combinatorial=False
    ),
    'Seg. fare': SegmentedParameterTuple(
        dict=all_segmentations, combinatorial=False
    ),
    'Seg. time': SegmentedParameterTuple(
        dict=all_segmentations, combinatorial=False
    ),
    'Seg. delay': SegmentedParameterTuple(
        dict=all_segmentations, combinatorial=False
    ),
    'Seg. legroom': SegmentedParameterTuple(
        dict=all_segmentations, combinatorial=False
    ),
}


## Step 8: Specification of the utility function. For each term, it is possible
## to define bounds on the coefficient, and to include a function that
## verifies its validity a posteriori.

utility_direct = [
    TermTuple(
        attribute='Fare direct',
        segmentation='Seg. fare',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Legroom direct',
        segmentation='Seg. legroom',
        bounds=(0, None),
        validity=None,
    ),
    TermTuple(
        attribute='Early direct',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Late direct',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Time direct',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
]

utility_same = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='Fare same',
        segmentation='Seg. fare',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Legroom same',
        segmentation='Seg. legroom',
        bounds=(0, None),
        validity=None,
    ),
    TermTuple(
        attribute='Early same',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Late same',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Time same',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
]

utility_multiple = [
    TermTuple(
        attribute=None,
        segmentation='Seg. cte',
        bounds=(None, None),
        validity=None,
    ),
    TermTuple(
        attribute='Fare multiple',
        segmentation='Seg. fare',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Legroom multiple',
        segmentation='Seg. legroom',
        bounds=(0, None),
        validity=None,
    ),
    TermTuple(
        attribute='Early multiple',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Late multiple',
        segmentation='Seg. delay',
        bounds=(None, 0),
        validity=None,
    ),
    TermTuple(
        attribute='Time multiple',
        segmentation='Seg. time',
        bounds=(None, 0),
        validity=None,
    ),
]

utilities = {
    1: ('Non stop', utility_direct),
    2: ('Same airline', utility_same),
    3: ('Multiple airlines', utility_multiple),
}

## Step 9: availabilities
availabilities = {1: 1, 2: 1, 3: 1}


# Step 10: We define potential candidates for the choice model.
def logit(V, av, choice):
    """logit model"""
    return models.loglogit(V, av, choice)


def nested1(V, av, choice):
    """Nested logit model: no stop / one stop"""
    onestop = Beta('mu_onestop', 1, 1, None, 0), [2, 3]
    nonstop = 1.0, [1]
    nests = nonstop, onestop
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    """Nested logit model: same / multiple"""
    same = Beta('mu_same', 1, 1, None, 0), [1, 2]
    multiple = 1.0, [3]
    nests = same, multiple
    return models.lognested(V, av, nests, choice)


def cnl1(V, av, choice):
    """Cross nested logit: fixed alphas"""
    mu_onestop = Beta('mu_onestop', 1, 1, None, 0)
    mu_same = Beta('mu_same', 1, 1, None, 0)
    alpha_onestop = {1: 1.0, 2: 0.5, 3: 1}
    alpha_same = {1: 1.0, 2: 0.5, 3: 1}
    nest_onestop = mu_onestop, alpha_onestop
    nest_same = mu_same, alpha_same
    nests = nest_onestop, nest_same
    return models.logcnl_avail(V, av, nests, choice)


def cnl2(V, av, choice):
    """Cross nested logit: estimated alphas"""
    alpha = Beta('alpha', 0.5, 0, 1, 0)
    mu_onestop = Beta('mu_onestop', 1, 1, None, 0)
    mu_same = Beta('mu_same', 1, 1, None, 0)
    alpha_onestop = {1: 1.0, 2: alpha, 3: 1}
    alpha_same = {1: 1.0, 2: 1 - alpha, 3: 1}
    nest_onestop = mu_onestop, alpha_onestop
    nest_same = mu_same, alpha_same
    nests = nest_onestop, nest_same
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
    'Airline',
    database,
    attributes,
    groupsOfAttributes,
    genericForbiden,
    forceActive,
    nonlinearSpecs,
    segmentations,
    utilities,
    availabilities,
    chosenAlternative,
    myModels,
)

theProblem.maximumNumberOfParameters = 300

# We propose several specifications to initialize the algorithm.
# For each group of attributes, we decide if it is nonlinear, and generic.
nl = {
    'Time': (5, False),
    'Fare': (0, False),
    'Legroom': (None, False),
    'Early': (None, False),
    'Late': (None, False),
}


# For each segmentation, we decided which dimensions are active.
sg = {'Seg. cte': ['TripPurpose'], 'Seg. legroom': ['Gender']}


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
