"""File process_data.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 11:01:21 2024

Import and process the data to be used in Biogeme

"""

import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Variable

# %
# Read the data file
df = pd.read_csv('data.csv')

# %
# Convert it to a Biogeme database.
database = Database('mdcev_example', df)

# %
# Associate each column with a Biogeme variable.
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
