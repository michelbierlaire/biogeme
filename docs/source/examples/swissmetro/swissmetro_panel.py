"""
.. _swissmetro_panel:

Panel data preparation for Swissmetro
=====================================

Data preparation for Swissmetro, and definition of the variables in
panel configuration

:author: Michel Bierlaire, EPFL
:date: Mon Mar  6 15:17:03 2023


"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# %%
# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# %%
# Definition of the variables.
PURPOSE = Variable('PURPOSE')
CHOICE = Variable('CHOICE')
GA = Variable('GA')
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
INCOME = Variable('INCOME')

# %%
# Removing some observations.
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# %%
# Definition of new variables
SM_COST = database.define_variable('SM_COST', SM_CO * (GA == 0))
TRAIN_COST = database.define_variable('TRAIN_COST', TRAIN_CO * (GA == 0))
CAR_AV_SP = database.define_variable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.define_variable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.define_variable('TRAIN_TT_SCALED', TRAIN_TT / 100)
TRAIN_COST_SCALED = database.define_variable('TRAIN_COST_SCALED', TRAIN_COST / 100)
SM_TT_SCALED = database.define_variable('SM_TT_SCALED', SM_TT / 100)
SM_COST_SCALED = database.define_variable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.define_variable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.define_variable('CAR_CO_SCALED', CAR_CO / 100)

# %%
# Qualify the data as panel. ID identifies the individuals.
database.panel('ID')

# %%
# We flatten the database, so that each row corresponds to one individual.
flat_df = database.generate_flat_panel_dataframe(identical_columns=None)
flat_database = db.Database('swissmetro_flat', flat_df)
