"""

Data preparation for Optima
===========================

Prepare data for the Optima case study.

:author: Michel Bierlaire
:date: Wed Apr 12 20:52:37 2023

"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# %%
# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# %%
# Variables from the data
Choice = Variable('Choice')
TimePT = Variable('TimePT')
TimeCar = Variable('TimeCar')
MarginalCostPT = Variable('MarginalCostPT')
CostCarCHF = Variable('CostCarCHF')
distance_km = Variable('distance_km')
Gender = Variable('Gender')
OccupStat = Variable('OccupStat')
Weight = Variable('Weight')

# %%
# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# %%
# Normalize the weights
sum_weight = database.data['Weight'].sum()
number_of_rows = database.data.shape[0]
normalized_weight = Weight * number_of_rows / sum_weight
