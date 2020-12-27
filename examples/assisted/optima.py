"""File optima.py

:author: Michel Bierlaire, EPFL
:date: Thu Dec 24 16:51:28 2020

Assisted specification for the Optima case study
"""

# Too constraining
# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.models as models
import biogeme.messaging as msg
import biogeme.vns as vns
import biogeme.assisted as assisted
from biogeme.expressions import (Beta,
                                 log,
                                 DefineVariable,
                                 Elem,
                                 Numeric,
                                 Variable)

logger = msg.bioMessage()
logger.setDebug()

# Read the data
df = pd.read_csv('optima.dat', '\t')

df.loc[df['OccupStat'] > 2, 'OccupStat'] = 3
df.loc[df['OccupStat'] == -1, 'OccupStat'] = 3

df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] == 5, 'Education'] = 4
df.loc[df['Education'] == 8, 'Education'] = 7

database = db.Database('optima', df)

globals().update(database.variables)

exclude = ((Choice == -1) + (CostCarCHF < 0) + (CarAvail == 3) * (Choice == 1)) > 0
database.remove(exclude)

# Definition of new variables


otherSubscription = DefineVariable('otherSubscription',
                                   ((HalfFareST == 1) +
                                    (LineRelST == 1) +
                                    (AreaRelST == 1) +
                                    (OtherST) == 1) > 0, database)

subscription = DefineVariable('subscription',
                              (GenAbST == 1) * 1 +
                              (GenAbST != 1) * otherSubscription * 2,
                              database)

TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5

# Definition of potential nonlinear transforms of variables
def mylog(x):
    """Log of the variable """
    return 'log', Elem({0:log(x), 1:Numeric(0)}, x == 0)

def sqrt(x):
    """Sqrt of the variable """
    return 'sqrt', x**0.5

def square(x):
    """Square of the variable """
    return 'square', x**2

def piecewise(x, thresholds, name):
    """Piecewise linear specification """
    piecewiseVariables = models.piecewiseVariables(x, thresholds)
    formula = piecewiseVariables[0]
    for k in range(1, len(thresholds)-1):
        formula += Beta(f'pw_{name}_{thresholds[k-1]}_{thresholds[k]}', 0, None, None, 0) * \
            piecewiseVariables[k]
    return (f'piecewise_{thresholds}', formula)

def piecewise_time_2(x):
    """ Piecewise linear for time :math:`0, 0.25, +\\infty` """
    return piecewise(x, [0, 0.25, None], 'time')

def piecewise_time_1(x):
    """ Piecewise linear for time :math:`0, 0.1, +\\infty` """
    return piecewise(x, [0, 0.1, None], 'time')

def piecewise_cost_2(x):
    """ Piecewise linear for cost :math:`0, 0.25, +\\infty` """
    return piecewise(x, [0, 0.25, None], 'cost')

def piecewise_cost_1(x):
    """ Piecewise linear for cost :math:`0, 0.1, +\\infty` """
    return piecewise(x, [0, 0.1, None], 'cost')


def boxcox(x, name):
    """ Box-Cox transform of the variable """
    ell = Beta(f'lambda_{name}', 1, 0.0001, 3.0, 0)
    return f'Box-Cox_{name}', models.boxcox(x, ell)

def boxcox_time(x):
    """ Box-Cox transform of the variable time """
    return boxcox(x, 'time')

def boxcox_cost(x):
    """ Box-Cox transform of the variable cost """
    return boxcox(x, 'cost')

def boxcox_headway(x):
    return boxcox(x, 'headway')

def distanceInteraction(x):
    return 'dist. interaction', x * log(1 + Variable('distance_km') / 1000)

# Define all possible segmentations
all_segmentations = {'TripPurpose': (TripPurpose, {1: 'work',
                                                   2: 'work_leisure',
                                                   3: 'leisure',
                                                   -1: 'missing'}),
                     'Urban': (UrbRur, {1: 'rural',
                                        2: 'urban'}),
                     'Language': (LangCode, {1: 'French', 2: 'German'}),
                     'Income': (Income, {1: '<2500',
                                         2: '2051_4000',
                                         3: '4001_6000',
                                         4: '6001_8000',
                                         5: '8001_10000',
                                         6: '>10000',
                                         -1: 'unknown'}),
                     'Gender': (Gender, {1: 'male',
                                         2: 'female',
                                         -1: 'unkown'}),
                     'Occupation': (OccupStat, {1: 'full_time',
                                                2: 'partial_time',
                                                3: 'others'}),
                     'Subscription': (subscription, {0: 'none',
                                                     1: 'GA',
                                                     2: 'other'}),
                     'CarAvail': (CarAvail, {1: 'always',
                                             2: 'sometimes',
                                             3: 'never',
                                             -1: 'unknown'}),
                     'Education': (Education, {3: 'vocational',
                                               4: 'high_school',
                                               6: 'higher_edu',
                                               7: 'university'})}


# Define segmentations
segmentations = {'Seg. cte': all_segmentations,
                 'Seg. cost': all_segmentations,
                 'Seg. wait': all_segmentations,
                 'Seg. time': all_segmentations,
                 'Seg. transfers': all_segmentations,
                 'Seg. dist': all_segmentations}

# Define the attributes of the alternatives
variables = {'PT travel time': TimePT_scaled,
             'PT travel cost': MarginalCostPT_scaled,
             'Car travel time': TimeCar_scaled,
             'Car travel cost': CostCarCHF_scaled,
             'Distance': distance_km_scaled,
             'Transfers': NbTransf,
             'PT Waiting time': WaitingTimePT}

# Group the attributes. All attributes in the same group will be
# assoaited with the same nonlinear transform, and the same
# segmentation. Attributes in the same group can be generic or
# alternative specific, except if mentioned otherwise
groupsOfVariables = {'Travel time': ['PT travel time',
                                     'Car travel time'],
                     'Travel cost': ['PT travel cost',
                                     'Car travel cost']}

# In this example, no variable must be alternative specific
genericForbiden = None

# In this example, all the variables must be in the model
forceActive = ['Travel time', 'Travel cost', 'Distance']

# Associate a list of potential nonlinearities with each group of variable
nonlinearSpecs = {'Travel time': [distanceInteraction,
                                  mylog,
                                  sqrt,
                                  square,
                                  boxcox_time],
                  'PT Waiting time': [mylog,
                                      sqrt,
                                      square,
                                      boxcox_time],
                  'Travel cost': [mylog,
                                  sqrt,
                                  square,
                                  boxcox_cost]}

# Specification of the utility function. For each term, it is possible
# to define bounds on the coefficient, and to include a function that
# verifies its validity a posteriori.

utility_pt = [(None, 'Seg. cte', (None, None), None),
              ('PT travel time', 'Seg. time', (None, 0), None),
              ('PT travel cost', 'Seg. cost', (None, 0), None),
              ('Transfers', 'Seg. transfers', (None, 0), None),
              ('PT Waiting time', 'Seg. wait', (None, 0), None)]


utility_car = [(None, 'Seg. cte', (None, None), None),
               ('Car travel time', 'Seg. time', (None, 0), None),
               ('Car travel cost', 'Seg. cost', (None, 0), None)]

utility_sm = [('Distance', 'Seg. dist', (None, 0), None)]


utilities = {0: ('pt', utility_pt),
             1: ('car', utility_car),
             2: ('sm', utility_sm)}

availabilities = {0: 1,
                  1: CarAvail != 3,
                  2: 1}


# We define potential candidates for the choice model.
def logit(V, av, choice):
    """ logit model """
    return models.loglogit(V, av, choice)

def nested1(V, av, choice):
    motorized = Beta('mu_motorized', 1, 1, None, 0), [0, 1]
    nonmotorized = 1.0, [2]
    nests = motorized, nonmotorized
    return models.lognested(V, av, nests, choice)


def nested2(V, av, choice):
    private = Beta('mu_private', 1, 1, None, 0), [1, 2]
    public = 1.0, [0]
    nests = private, public
    return models.lognested(V, av, nests, choice)


def cnl1(V, av, choice):
    """ Cross nested logit: fixed alphas """
    mu_motorized = Beta('mu_motorized', 1, 1, None, 0)
    mu_public = Beta('mu_public', 1, 1, None, 0)
    alpha_motorized = {0: 1.0, 1: 0.5, 2:0}
    alpha_public = {0: 0, 1: 0.5, 2:1}
    nest_motorized = mu_motorized, alpha_motorized
    nest_public = mu_public, alpha_public
    nests = nest_motorized, nest_public
    return models.logcnl_avail(V, av, nests, choice)

def cnl2(V, av, choice):
    """ Cross nested logit: fixed alphas """
    alpha = Beta('alpha', 0.5, 0, 1, 0)
    mu_motorized = Beta('mu_motorized', 1, 1, None, 0)
    mu_public = Beta('mu_public', 1, 1, None, 0)
    alpha_motorized = {0: 1.0, 1: alpha, 2:0}
    alpha_public = {0: 0, 1: 1-alpha, 2:1}
    nest_motorized = mu_motorized, alpha_motorized
    nest_public = mu_public, alpha_public
    nests = nest_motorized, nest_public
    return models.logcnl_avail(V, av, nests, choice)

# We provide names to these candidates
myModels = {'Logit': logit,
            'Nested one stop': nested1,
            'Nested same': nested2,
            'CNL alpha fixed': cnl1,
            'CNL alpha est.': cnl2}

# Definition of the specification problem, gathering all information defined above.
theProblem = assisted.specificationProblem('Optima',
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
                                           myModels)

theProblem.maximumNumberOfParameters = 100

# We propose several specifications to initialize the algorithm.
# For each group of attributes, we decide if it is nonlinear, and generic.
nl1 = {'Travel time': (2, False),
       'Travel cost': (None, False),
       'Distance': (None, False)}

nl2 = {'Travel time': (None, True),
       'Travel cost': (3, False),
       'Distance': (None, False),
       'Transfers': (None, False),
       'PT Waiting time': (0, False)}



# For each segmentation, we decided which dimensions are active.
sg1 = {'Seg. cte': ['CarAvail'],
       'Seg. cost': ['Income'],
       'Seg. dist': ['Urban']}



initSolutions = [
     theProblem.generateSolution(nl1, sg1, 'Logit'),
     theProblem.generateSolution(nl1, sg1, 'Nested one stop'),
     theProblem.generateSolution(nl1, sg1, 'Nested same'),
     theProblem.generateSolution(nl1, sg1, 'CNL alpha fixed'),
     theProblem.generateSolution(nl1, sg1, 'CNL alpha est.'),
     theProblem.generateSolution(nl2, sg1, 'Logit'),
     theProblem.generateSolution(nl2, sg1, 'Nested one stop'),
     theProblem.generateSolution(nl2, sg1, 'Nested same'),
     theProblem.generateSolution(nl2, sg1, 'CNL alpha fixed'),
     theProblem.generateSolution(nl2, sg1, 'CNL alpha est.')]

# Optimization algorithm
vns.vns(theProblem,
        initSolutions,
        archiveInputFile='optimaPareto.pickle',
        pickleOutputFile='optimaPareto.pickle')
