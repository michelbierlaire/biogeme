"""File 02simulation.py

:author: Michel Bierlaire, EPFL
:date: Thu Oct 28 15:10:46 2021

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We simulate various formulas with the estimated model
"""
import sys
import time
import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
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

# Normalize the weights
sumWeight = database.data['Weight'].sum()
numberOfRows = database.data.shape[0]
normalizedWeight = Weight * numberOfRows / sumWeight

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Define new variables. Must be consistent with estimation results.
TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5
male = Gender == 1
female = Gender == 2
unreportedGender = Gender == -1
fulltime = OccupStat == 1
notfulltime = OccupStat != 1

# Normalize the weights
sumWeight = database.data['Weight'].sum()
numberOfRows = database.data.shape[0]
normalizedWeight = Weight * numberOfRows / sumWeight


MarginalCostScenario = MarginalCostPT
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
V_PT = (
    ASC_PT
    + BETA_TIME_FULLTIME * TimePT_scaled * fulltime
    + BETA_TIME_OTHER * TimePT_scaled * notfulltime
    + BETA_COST * MarginalCostPT_scaled
)
V_CAR = (
    ASC_CAR
    + BETA_TIME_FULLTIME * TimeCar_scaled * fulltime
    + BETA_TIME_OTHER * TimeCar_scaled * notfulltime
    + BETA_COST * CostCarCHF_scaled
)
V_SM = (
    ASC_SM
    + BETA_DIST_MALE * distance_km_scaled * male
    + BETA_DIST_FEMALE * distance_km_scaled * female
    + BETA_DIST_UNREPORTED * distance_km_scaled * unreportedGender
)
V = {0: V_PT, 1: V_CAR, 2: V_SM}
MU_NOCAR = Beta('MU_NOCAR', 1.0, 1.0, None, 0)
CAR_NEST = 1.0, [1]
NO_CAR_NEST = MU_NOCAR, [0, 2]
nests = CAR_NEST, NO_CAR_NEST
prob_pt = models.nested(V, None, nests, 0)
prob_car = models.nested(V, None, nests, 1)
prob_sm = models.nested(V, None, nests, 2)
prob_chosen = models.nested(V, None, nests, Choice)


simulate = {
    'weight': normalizedWeight,
    'Utility PT': V_PT,
    'Utility car': V_CAR,
    'Utility SM': V_SM,
    'Prob. PT': prob_pt,
    'Prob. car': prob_car,
    'Prob. SM': prob_sm,
    'Prob. chosen': prob_chosen,
}

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='01nestedEstimation.pickle')
except FileNotFoundError:
    sys.exit(
        'Run first the script 01nestedEstimation.py in order to generate '
        'the file 01nestedEstimation.pickle.'
    )

# Simulate the expressions
start_time = time.time()
biogeme = bio.BIOGEME(database, simulate)
biogeme.modelName = '02simulation'
simulatedValues = biogeme.simulate(results.getBetaValues())
print(f'--- Execution time with Biogeme: {time.time() - start_time} seconds ---')


# It can also be done without using the Biogeme object. In that case,
# each expression is simulated indendently. Not only it is more
# cumbersome to code, but it may take more time.

start_time = time.time()
simulate_formulas = {
    'weight': normalizedWeight.getValue_c(betas=results.getBetaValues(), database=database),
    'Utility PT': V_PT.getValue_c(betas=results.getBetaValues(), database=database),
    'Utility car': V_CAR.getValue_c(betas=results.getBetaValues(), database=database),
    'Utility SM': V_SM.getValue_c(betas=results.getBetaValues(), database=database),
    'Prob. PT': prob_pt.getValue_c(betas=results.getBetaValues(), database=database),
    'Prob. car': prob_car.getValue_c(betas=results.getBetaValues(), database=database),
    'Prob. SM': prob_sm.getValue_c(betas=results.getBetaValues(), database=database),
    'Prob. chosen': prob_chosen.getValue_c(betas=results.getBetaValues(), database=database),
}

another_simulated_values = pd.DataFrame.from_dict(
    simulate_formulas,
)
print(f'--- Execution time without Biogeme: {time.time() - start_time} seconds ---')

print(simulatedValues)
print(another_simulated_values)

# We can check the log likelihood

print(f'Log likelihood from simulation: {np.sum(np.log(simulatedValues["Prob. chosen"]))}')
print(f'Log likelihood from estimation: {results.data.logLike}')

# Market shares are calculated using the weighted mean of the individual probabilities
market_share_pt = np.mean(simulatedValues['weight'] * simulatedValues['Prob. PT'])
market_share_car = np.mean(simulatedValues['weight'] * simulatedValues['Prob. car'])
market_share_SM = np.mean(simulatedValues['weight'] * simulatedValues['Prob. SM'])
print(f'Public transportation: {100*market_share_pt:.3g}%')
print(f'Car                  : {100*market_share_car:.3g}%')
print(f'Slow modes           : {100*market_share_SM:.3g}%')
