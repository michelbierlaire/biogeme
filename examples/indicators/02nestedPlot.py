"""File 02nestedPlot.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 11 10:15:18 2019

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We simulate pricing scenarios and their impact on the revenues.
"""

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta

# Library for plotting
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Define new variables. Must be consistent with estimation results.
TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5
male = (Gender == 1)
female = (Gender == 2)
unreportedGender = (Gender == -1)
fulltime = (OccupStat == 1)
notfulltime = (OccupStat != 1)

# Normalize the weights
sumWeight = database.data['Weight'].sum()
numberOfRows = database.data.shape[0]
normalizedWeight = Weight * numberOfRows / sumWeight

def scenario(scale):
    """Simulate a scenarios modifying the price of public transportation

    :param scale: price multiplier.
    :type scale: float

    :return: simulated revenues
    :rtype: float
    """
    # This is the only variable that depends on scale
    MarginalCostScenario = MarginalCostPT * scale
    MarginalCostPT_scaled = MarginalCostScenario / 10

    # The rest of the model is the same for all scenarios
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 1)
    ASC_SM = Beta('ASC_SM', 0, None, None, 0)
    BETA_TIME_FULLTIME = Beta('BETA_TIME_FULLTIME', 0, None, None, 0)
    BETA_TIME_OTHER = Beta('BETA_TIME_OTHER', 0, None, None, 0)
    BETA_DIST_MALE = Beta('BETA_DIST_MALE', 0, None, None, 0)
    BETA_DIST_FEMALE = Beta('BETA_DIST_FEMALE', 0, None, None, 0)
    BETA_DIST_UNREPORTED = Beta('BETA_DIST_UNREPORTED', 0, None, None, 0)
    BETA_COST = Beta('BETA_COST', 0, None, None, 0)
    # Utility functions
    V_PT = ASC_PT + BETA_TIME_FULLTIME * TimePT_scaled * fulltime + \
        BETA_TIME_OTHER * TimePT_scaled * notfulltime + \
        BETA_COST * MarginalCostPT_scaled
    V_CAR = ASC_CAR + \
        BETA_TIME_FULLTIME * TimeCar_scaled * fulltime + \
        BETA_TIME_OTHER * TimeCar_scaled * notfulltime + \
        BETA_COST * CostCarCHF_scaled
    V_SM = ASC_SM + \
        BETA_DIST_MALE * distance_km_scaled * male + \
        BETA_DIST_FEMALE * distance_km_scaled * female + \
        BETA_DIST_UNREPORTED * distance_km_scaled * unreportedGender
    V = {0: V_PT,
         1: V_CAR,
         2: V_SM}
    MU_NOCAR = Beta('MU_NOCAR', 1.0, 1.0, None, 0)
    CAR_NEST = 1.0, [1]
    NO_CAR_NEST = MU_NOCAR, [0, 2]
    nests = CAR_NEST, NO_CAR_NEST
    prob_pt = models.nested(V, None, nests, 0)
    simulate = {'weight': normalizedWeight,
                'Revenue public transportation':
                prob_pt * MarginalCostScenario}

    biogeme = bio.BIOGEME(database, simulate)
    biogeme.modelName = '02nestedPlot'

    # Read the estimation results from the file
    try:
        results = res.bioResults(pickleFile='01nestedEstimation.pickle')
    except FileNotFoundError:
        sys.exit('Run first the script 01nestedEstimation.py in order to generate '
                 'the file 01nestedEstimation.pickle.')
    # Simulation
    simulatedValues = biogeme.simulate(results.getBetaValues())

    # We calculate the sum for all individuals of the generated revenues.
    revenues_pt = (simulatedValues['Revenue public transportation'] *
                   simulatedValues['weight']).sum()
    return revenues_pt

# We now plot the relationship between the cost and the revenues
scales = np.arange(0.0, 5.0, 0.1)
revenues = [scenario(s) for s in scales]
plt.plot(scales, revenues)
plt.xlabel('Modification of the price of public transportation (%)')
plt.ylabel('Revenues')
plt.show()
