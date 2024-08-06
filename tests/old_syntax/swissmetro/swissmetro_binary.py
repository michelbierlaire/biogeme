"""
.. _swissmetro_binary:

Data preparation for Swissmetro (binary choice)
===============================================

Data preparation for Swissmetro, and definition of the variables. The
data is designed to estimate binary logit models. All observations
such that Swissmetro was chosen are removed.

:author: Michel Bierlaire, EPFL
:date: Mon Mar  6 15:17:03 2023
"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# %%
# Read the data.
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# %%
# Definition of the variables.
PURPOSE = Variable('PURPOSE')
CHOICE = Variable('CHOICE')
GA = Variable('GA')
LUGGAGE = Variable('LUGGAGE')
TRAIN_CO = Variable('TRAIN_CO')
CAR_AV = Variable('CAR_AV')
SP = Variable('SP')
TRAIN_AV = Variable('TRAIN_AV')
TRAIN_TT = Variable('TRAIN_TT')
SM_TT = Variable('SM_TT')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
SM_CO = Variable('SM_CO')
SM_AV = Variable('SM_AV')
MALE = Variable('MALE')
GROUP = Variable('GROUP')
TRAIN_HE = Variable('TRAIN_HE')
SM_HE = Variable('SM_HE')
INCOME = Variable('INCOME')

# Excluding observations.
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0) + (CHOICE == 2)) > 0
database.remove(exclude)

# %%
# Definition of new variables.
SM_COST = database.DefineVariable('SM_COST', SM_CO * (GA == 0))
TRAIN_COST = database.DefineVariable('TRAIN_COST', TRAIN_CO * (GA == 0))
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100)
TRAIN_COST_SCALED = database.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)
