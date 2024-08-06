"""

Data preparation for Swissmetro: one observation
================================================

Use only the first observation for simulation.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 20:51:58 2023
"""
import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# %%
df = pd.read_csv('swissmetro.dat', sep='\t')

# %%
# Use only the first observation (index 0)
df = df.drop(df[df.index != 0].index)
database = db.Database('swissmetro', df)

# %%
SM_CO = Variable('SM_CO')
TRAIN_CO = Variable('TRAIN_CO')
CAR_CO = Variable('CAR_CO')
TRAIN_TT = Variable('TRAIN_TT')
SM_TT = Variable('SM_TT')
CAR_TT = Variable('CAR_TT')
GA = Variable('GA')
CAR_AV = Variable('CAR_AV')
TRAIN_AV = Variable('TRAIN_AV')
SM_AV = Variable('SM_AV')
SP = Variable('SP')
CHOICE = Variable('CHOICE')

# %%
# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100.0
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100.0
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100
