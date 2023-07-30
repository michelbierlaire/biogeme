"""File swissmetro_data.py

:author: Michel Bierlaire, EPFL
:date: Mon Mar  6 15:17:03 2023

Data preparation for Swissmetro, and definition of the variables
"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

GROUP = Variable('GROUP')
SURVEY = Variable('SURVEY')
SP = Variable('SP')
ID = Variable('ID')
PURPOSE = Variable('PURPOSE')
FIRST = Variable('FIRST')
TICKET = Variable('TICKET')
WHO = Variable('WHO')
LUGGAGE = Variable('LUGGAGE')
AGE = Variable('AGE')
MALE = Variable('MALE')
INCOME = Variable('INCOME')
GA = Variable('GA')
ORIGIN = Variable('ORIGIN')
DEST = Variable('DEST')
TRAIN_AV = Variable('TRAIN_AV')
CAR_AV = Variable('CAR_AV')
SM_AV = Variable('SM_AV')
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_CO = Variable('TRAIN_CO')
TRAIN_HE = Variable('TRAIN_HE')
SM_TT = Variable('SM_TT')
SM_CO = Variable('SM_CO')
SM_HE = Variable('SM_HE')
SM_SEATS = Variable('SM_SEATS')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
CHOICE = Variable('CHOICE')

# Removing some observations can be done directly using pandas.
# remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
# database.data.drop(database.data[remove].index,inplace=True)
# Here we use the "biogeme" way:
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)


# Definition of new variables
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
