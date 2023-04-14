"""File optima.py

Data processing for the optima case study.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 16:40:59 2023

"""


import pandas as pd
import biogeme.database as db
from biogeme import models
from biogeme.expressions import Variable


# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

Choice = Variable('Choice')
TimePT = Variable('TimePT')
MarginalCostPT = Variable('MarginalCostPT')
TimeCar = Variable('TimeCar')
CostCarCHF = Variable('CostCarCHF')
distance_km = Variable('distance_km')
TripPurpose = Variable('TripPurpose')
WaitingTimePT = Variable('WaitingTimePT')
CalculatedIncome = Variable('CalculatedIncome')
age = Variable('age')
NbCar = Variable('NbCar')
NbBicy = Variable('NbBicy')
HouseType = Variable('HouseType')
Gender = Variable('Gender')
FamilSitu = Variable('FamilSitu')
GenAbST = Variable('GenAbST')
Education = Variable('Education')
Envir01 = Variable('Envir01')
Envir02 = Variable('Envir02')
Envir03 = Variable('Envir03')
Mobil11 = Variable('Mobil11')
Mobil14 = Variable('Mobil14')
Mobil16 = Variable('Mobil16')
Mobil17 = Variable('Mobil17')


# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Piecewise linear definition of income
ScaledIncome = database.DefineVariable('ScaledIncome', CalculatedIncome / 1000)

thresholds = [None, 4, 6, 8, 10, None]
formulaIncome = models.piecewiseFormula(
    ScaledIncome, thresholds, [0.0, 0.0, 0.0, 0.0, 0.0]
)

# Definition of other variables
age_65_more = database.DefineVariable('age_65_more', age >= 65)
moreThanOneCar = database.DefineVariable('moreThanOneCar', NbCar > 1)
moreThanOneBike = database.DefineVariable('moreThanOneBike', NbBicy > 1)
individualHouse = database.DefineVariable('individualHouse', HouseType == 1)
male = database.DefineVariable('male', Gender == 1)
haveChildren = database.DefineVariable(
    'haveChildren', ((FamilSitu == 3) + (FamilSitu == 4)) > 0
)
haveGA = database.DefineVariable('haveGA', GenAbST == 1)
highEducation = database.DefineVariable('highEducation', Education >= 6)
