"""

Data processing for the optima case study
=========================================

Read the data from file, and define variables.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 16:40:59 2023

"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# %%
# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# %%
ID = Variable('ID')
DestAct = Variable('DestAct')
NbTransf = Variable('NbTransf')
TimePT = Variable('TimePT')
WalkingTimePT = Variable('WalkingTimePT')
WaitingTimePT = Variable('WaitingTimePT')
CostPT = Variable('CostPT')
CostCar = Variable('CostCar')
TimeCar = Variable('TimeCar')
NbHousehold = Variable('NbHousehold')
NbChild = Variable('NbChild')
NbCar = Variable('NbCar')
NbMoto = Variable('NbMoto')
NbBicy = Variable('NbBicy')
NbBicyChild = Variable('NbBicyChild')
NbComp = Variable('NbComp')
NbTV = Variable('NbTV')
Internet = Variable('Internet')
NewsPaperSubs = Variable('NewsPaperSubs')
NbCellPhones = Variable('NbCellPhones')
NbSmartPhone = Variable('NbSmartPhone')
HouseType = Variable('HouseType')
OwnHouse = Variable('OwnHouse')
NbRoomsHouse = Variable('NbRoomsHouse')
YearsInHouse = Variable('YearsInHouse')
Income = Variable('Income')
Gender = Variable('Gender')
BirthYear = Variable('BirthYear')
Mothertongue = Variable('Mothertongue')
FamilSitu = Variable('FamilSitu')
OccupStat = Variable('OccupStat')
SocioProfCat = Variable('SocioProfCat')
CalculatedIncome = Variable('CalculatedIncome')
Education = Variable('Education')
HalfFareST = Variable('HalfFareST')
LineRelST = Variable('LineRelST')
GenAbST = Variable('GenAbST')
AreaRelST = Variable('AreaRelST')
OtherST = Variable('OtherST')
CarAvail = Variable('CarAvail')
MarginalCostPT = Variable('MarginalCostPT')
CostCarCHF = Variable('CostCarCHF')
Envir01 = Variable('Envir01')
Envir02 = Variable('Envir02')
Envir03 = Variable('Envir03')
Envir04 = Variable('Envir04')
Envir05 = Variable('Envir05')
Envir06 = Variable('Envir06')
Mobil01 = Variable('Mobil01')
Mobil02 = Variable('Mobil02')
Mobil03 = Variable('Mobil03')
Mobil04 = Variable('Mobil04')
Mobil05 = Variable('Mobil05')
Mobil06 = Variable('Mobil06')
Mobil07 = Variable('Mobil07')
Mobil08 = Variable('Mobil08')
Mobil09 = Variable('Mobil09')
Mobil10 = Variable('Mobil10')
Mobil11 = Variable('Mobil11')
Mobil12 = Variable('Mobil12')
Mobil13 = Variable('Mobil13')
Mobil14 = Variable('Mobil14')
Mobil15 = Variable('Mobil15')
Mobil16 = Variable('Mobil16')
Mobil17 = Variable('Mobil17')
Mobil18 = Variable('Mobil18')
Mobil19 = Variable('Mobil19')
Mobil20 = Variable('Mobil20')
Mobil21 = Variable('Mobil21')
Mobil22 = Variable('Mobil22')
Mobil23 = Variable('Mobil23')
Mobil24 = Variable('Mobil24')
Mobil25 = Variable('Mobil25')
Mobil26 = Variable('Mobil26')
Mobil27 = Variable('Mobil27')
ResidCh01 = Variable('ResidCh01')
ResidCh02 = Variable('ResidCh02')
ResidCh03 = Variable('ResidCh03')
ResidCh04 = Variable('ResidCh04')
ResidCh05 = Variable('ResidCh05')
ResidCh06 = Variable('ResidCh06')
ResidCh07 = Variable('ResidCh07')
LifSty01 = Variable('LifSty01')
LifSty02 = Variable('LifSty02')
LifSty03 = Variable('LifSty03')
LifSty04 = Variable('LifSty04')
LifSty05 = Variable('LifSty05')
LifSty06 = Variable('LifSty06')
LifSty07 = Variable('LifSty07')
LifSty08 = Variable('LifSty08')
LifSty09 = Variable('LifSty09')
LifSty10 = Variable('LifSty10')
LifSty11 = Variable('LifSty11')
LifSty12 = Variable('LifSty12')
LifSty13 = Variable('LifSty13')
LifSty14 = Variable('LifSty14')
TripPurpose = Variable('TripPurpose')
TypeCommune = Variable('TypeCommune')
UrbRur = Variable('UrbRur')
LangCode = Variable('LangCode')
ClassifCodeLine = Variable('ClassifCodeLine')
frequency = Variable('frequency')
ResidChild = Variable('ResidChild')
NbTrajects = Variable('NbTrajects')
FreqCarPar = Variable('FreqCarPar')
FreqTrainPar = Variable('FreqTrainPar')
FreqOtherPar = Variable('FreqOtherPar')
FreqTripHouseh = Variable('FreqTripHouseh')
Region = Variable('Region')
distance_km = Variable('distance_km')
Choice = Variable('Choice')
InVehicleTime = Variable('InVehicleTime')
ModeToSchool = Variable('ModeToSchool')
ReportedDuration = Variable('ReportedDuration')
CoderegionCAR = Variable('CoderegionCAR')
age = Variable('age')
Weight = Variable('Weight')

# %%
# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# %%
# Definition of other variables
ScaledIncome = database.DefineVariable('ScaledIncome', CalculatedIncome / 1000)
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
childCenter = database.DefineVariable(
    'childCenter', ((ResidChild == 1) + (ResidChild == 2)) > 0
)

childSuburb = database.DefineVariable(
    'childSuburb', ((ResidChild == 3) + (ResidChild == 4)) > 0
)
TimePT_scaled = database.DefineVariable('TimePT_scaled', TimePT / 200)
TimeCar_scaled = database.DefineVariable('TimeCar_scaled', TimeCar / 200)
MarginalCostPT_scaled = database.DefineVariable(
    'MarginalCostPT_scaled', MarginalCostPT / 10
)
CostCarCHF_scaled = database.DefineVariable('CostCarCHF_scaled', CostCarCHF / 10)
distance_km_scaled = database.DefineVariable('distance_km_scaled', distance_km / 5)
PurpHWH = database.DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = database.DefineVariable('PurpOther', TripPurpose != 1)
