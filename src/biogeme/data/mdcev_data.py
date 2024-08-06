"""File mengyi_data.py

:author: Michel Bierlaire, EPFL
:date: 

First attempt to estimate a MDCEV model

"""

import os

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable


def read_data() -> db.Database:
    """Read the data from file"""
    # Get the directory of the current file
    module_dir = os.path.dirname(__file__)

    # Construct the path to the data file
    data_file_path = os.path.join(module_dir, 'data', 'mdcev.csv')

    # %%
    # Read the data.
    df = pd.read_csv(data_file_path, sep='\t')
    database = db.Database('mdcev', df)
    return database


PersonID = Variable('PersonID')
weight = Variable('weight')
hhsize = Variable('hhsize')
childnum = Variable('childnum')
faminc = Variable('faminc')
faminc25K = Variable('faminc25K')
income = Variable('income')
employed = Variable('employed')
fulltime = Variable('fulltime')
spousepr = Variable('spousepr')
spousemp = Variable('spousemp')
male = Variable('male')
married = Variable('married')
age = Variable('age')
age2 = Variable('age2')
age15_40 = Variable('age15_40')
age41_60 = Variable('age41_60')
age61_85 = Variable('age61_85')
bachigher = Variable('bachigher')
white = Variable('white')
metro = Variable('metro')
diaryday = Variable('diaryday')
Sunday = Variable('Sunday')
holiday = Variable('holiday')
weekearn = Variable('weekearn')
weekwordur = Variable('weekwordur')
hhchild = Variable('hhchild')
ohhchild = Variable('ohhchild')
t1 = Variable('t1')
t2 = Variable('t2')
t3 = Variable('t3')
t4 = Variable('t4')
number_chosen = Variable('number_chosen')
