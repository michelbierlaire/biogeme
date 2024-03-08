"""
.. _swissmetro_data:

Data preparation for Swissmetro
===============================

Data preparation and definition of the variables.

:author: Michel Bierlaire, EPFL
:date: Mon Mar  6 15:17:03 2023

"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

import os


def read_data() -> db.Database:
    """Read the data from file"""
    # Get the directory of the current file
    module_dir = os.path.dirname(__file__)

    # Construct the path to the data file
    data_file_path = os.path.join(module_dir, 'data', 'swissmetro.dat')

    # %%
    # Read the data.
    df = pd.read_csv(data_file_path, sep='\t')
    database = db.Database('swissmetro', df)
    exclude = CHOICE == 0
    # exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    database.remove(exclude)
    # Definition of new variables.
    _ = database.define_variable('SM_COST', SM_CO * (GA == 0))
    _ = database.define_variable('TRAIN_COST', TRAIN_CO * (GA == 0))
    _ = database.define_variable('CAR_AV_SP', CAR_AV * (SP != 0))
    _ = database.define_variable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
    _ = database.define_variable('TRAIN_TT_SCALED', TRAIN_TT / 100)
    _ = database.define_variable('TRAIN_COST_SCALED', TRAIN_COST / 100)
    _ = database.define_variable('SM_TT_SCALED', SM_TT / 100)
    _ = database.define_variable('SM_COST_SCALED', SM_COST / 100)
    _ = database.define_variable('CAR_TT_SCALED', CAR_TT / 100)
    _ = database.define_variable('CAR_CO_SCALED', CAR_CO / 100)
    return database


# Definition of the variables.
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
SM_COST = Variable('SM_COST')
TRAIN_COST = Variable('TRAIN_COST')
CAR_AV_SP = Variable('CAR_AV_SP')
TRAIN_AV_SP = Variable('TRAIN_AV_SP')
TRAIN_TT_SCALED = Variable('TRAIN_TT_SCALED')
TRAIN_COST_SCALED = Variable('TRAIN_COST_SCALED')
SM_TT_SCALED = Variable('SM_TT_SCALED')
SM_COST_SCALED = Variable('SM_COST_SCALED')
CAR_TT_SCALED = Variable('CAR_TT_SCALED')
CAR_CO_SCALED = Variable('CAR_CO_SCALED')
