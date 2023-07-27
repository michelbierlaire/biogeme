"""File optima_data.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 11:26:56 2023

Data for the Optima case study
"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# Read the data
df = pd.read_csv('optima.dat', sep='\t')

df.loc[df['OccupStat'] > 2, 'OccupStat'] = 3
df.loc[df['OccupStat'] == -1, 'OccupStat'] = 3

df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] <= 3, 'Education'] = 3
df.loc[df['Education'] == 5, 'Education'] = 4
df.loc[df['Education'] == 8, 'Education'] = 7

df.loc[df['TripPurpose'] != 1, 'TripPurpose'] = 2

df.loc[df['CarAvail'] != 3, 'CarAvail'] = 1

database = db.Database('optima', df)

Choice = Variable('Choice')
CostCarCHF = Variable('CostCarCHF')
CarAvail = Variable('CarAvail')
HalfFareST = Variable('HalfFareST')
LineRelST = Variable('LineRelST')
AreaRelST = Variable('AreaRelST')
OtherST = Variable('OtherST')
GenAbST = Variable('GenAbST')
TimePT = Variable('TimePT')
TimeCar = Variable('TimeCar')
MarginalCostPT = Variable('MarginalCostPT')
distance_km = Variable('distance_km')
WaitingTimePT = Variable('WaitingTimePT')
TripPurpose = Variable('TripPurpose')
UrbRur = Variable('UrbRur')
LangCode = Variable('LangCode')
Gender = Variable('Gender')
OccupStat = Variable('OccupStat')
subscription = Variable('subscription')
CarAvail = Variable('CarAvail')
Education = Variable('Education')
NbTransf = Variable('NbTransf')
age = Variable('age')

exclude = ((Choice == -1) + (CostCarCHF < 0) + (CarAvail == 3) * (Choice == 1)) > 0
database.remove(exclude)

# Definition of new variables

otherSubscription = database.DefineVariable(
    'otherSubscription',
    ((HalfFareST == 1) + (LineRelST == 1) + (AreaRelST == 1) + (OtherST) == 1) > 0,
)

subscription = database.DefineVariable(
    'subscription', (GenAbST == 1) * 1 + (GenAbST != 1) * otherSubscription * 2
)

TimePT_scaled = database.DefineVariable('TimePT_scaled', TimePT / 200)
TimeCar_scaled = database.DefineVariable('TimeCar_scaled', TimeCar / 200)
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5
