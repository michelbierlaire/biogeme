"""
.. _optima_data:

Data preparation for Optima
===========================

Prepare data for the Optima case study.

:author: Michel Bierlaire
:date: Wed Apr 12 20:52:37 2023

"""

import os

import biogeme.database as db
import pandas as pd
from biogeme.expressions import Variable

# Get the directory of the current file
module_dir = os.path.dirname(__file__)
# Construct the path to the data file
data_file_path = os.path.join(module_dir, 'data', 'optima.dat')


# %%
# Read the data
def read_data() -> db.Database:
    """Read the data from file"""
    df = pd.read_csv(data_file_path, sep='\t')
    # Exclude observations such that the chosen alternative is -1
    df.drop(df[df['Choice'] == -1].index, inplace=True)

    car_not_available = df['CarAvail'] == 3
    car_is_chosen = df['Choice'] == 1
    incompatible = car_is_chosen & car_not_available
    df.drop(df[incompatible].index, inplace=True)

    # Normalize the weights
    sum_weight = df['Weight'].sum()
    number_of_rows = df.shape[0]
    df['normalized_weight'] = df['Weight'] * number_of_rows / sum_weight
    database = db.Database(name=data_file_path, dataframe=df)
    _ = database.define_variable('livesInUrbanArea', UrbRur == 2)
    _ = database.define_variable('owningHouse', OwnHouse == 1)
    _ = database.define_variable('ScaledIncome', CalculatedIncome / 1000)
    _ = database.define_variable('age_65_more', age >= 65)
    _ = database.define_variable('age_30_less', age <= 30)
    _ = database.define_variable('age_category', 2 - (age <= 30) + (age >= 65))
    _ = database.define_variable(
        'household_size', 3 - 2 * (NbHousehold == 1) - (NbHousehold == 2)
    )
    # 15% / 50% / 85 % quantiles of the income distribution in the data
    _ = database.define_variable(
        'income_category',
        1
        + (CalculatedIncome >= 3250)
        + (CalculatedIncome >= 7000)
        + (CalculatedIncome >= 15000),
    )

    _ = database.define_variable('moreThanOneCar', NbCar > 1)
    _ = database.define_variable('moreThanOneBike', NbBicy > 1)
    _ = database.define_variable('individualHouse', HouseType == 1)
    _ = database.define_variable('male', Gender == 1)
    _ = database.define_variable('single', ((FamilSitu == 1) + (FamilSitu == 4)) > 0)
    _ = database.define_variable(
        'haveChildren', ((FamilSitu == 3) + (FamilSitu == 4)) > 0
    )
    _ = database.define_variable('haveGA', GenAbST == 1)
    _ = database.define_variable('highEducation', Education >= 6)
    _ = database.define_variable('artisans', SocioProfCat == 5)
    _ = database.define_variable('employees', SocioProfCat == 6)
    _ = database.define_variable(
        'occupation_status',
        4 - 3 * (OccupStat == 1) - 2 * (OccupStat == 2) - (OccupStat == 9),
    )
    _ = database.define_variable(
        'childCenter', ((ResidChild == 1) + (ResidChild == 2)) > 0
    )

    _ = database.define_variable(
        'childSuburb', ((ResidChild == 3) + (ResidChild == 4)) > 0
    )
    _ = database.define_variable('TimePT_scaled', TimePT / 200)
    _ = database.define_variable('TimePT_hour', TimePT / 60)
    _ = database.define_variable('TimeCar_scaled', TimeCar / 200)
    _ = database.define_variable('TimeCar_hour', TimeCar / 60)
    _ = database.define_variable('MarginalCostPT_scaled', MarginalCostPT / 10)
    _ = database.define_variable('CostCarCHF_scaled', CostCarCHF / 10)
    _ = database.define_variable('distance_km_scaled', distance_km / 5)
    _ = database.define_variable('PurpHWH', TripPurpose == 1)
    _ = database.define_variable('PurpOther', TripPurpose != 1)
    _ = database.define_variable(
        'number_of_trips',
        (NbTrajects == 1) + 2 * (NbTrajects == 2) + 3 * (NbTrajects >= 3),
    )
    # urbanization: 1 = urban, 2 = mixed, 3 = rural
    _ = database.define_variable(
        'urbanization', 2 - 1 * (TypeCommune <= 3) + 1 * (TypeCommune >= 8)
    )

    return database


# %%
# Variables from the data
Choice = Variable('Choice')
TimePT = Variable('TimePT')
TimeCar = Variable('TimeCar')
MarginalCostPT = Variable('MarginalCostPT')
CostCarCHF = Variable('CostCarCHF')
distance_km = Variable('distance_km')
Gender = Variable('Gender')
OccupStat = Variable('OccupStat')
Weight = Variable('Weight')
ID = Variable('ID')
DestAct = Variable('DestAct')
NbTransf = Variable('NbTransf')
WalkingTimePT = Variable('WalkingTimePT')
WaitingTimePT = Variable('WaitingTimePT')
CostPT = Variable('CostPT')
CostCar = Variable('CostCar')
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
BirthYear = Variable('BirthYear')
Mothertongue = Variable('Mothertongue')
FamilSitu = Variable('FamilSitu')
SocioProfCat = Variable('SocioProfCat')
CalculatedIncome = Variable('CalculatedIncome')
Education = Variable('Education')
HalfFareST = Variable('HalfFareST')
LineRelST = Variable('LineRelST')
GenAbST = Variable('GenAbST')
AreaRelST = Variable('AreaRelST')
OtherST = Variable('OtherST')
urbanization = Variable('urbanization')
three_trips_or_more = Variable('three_trips_or_more')
CarAvail = Variable('CarAvail')
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
InVehicleTime = Variable('InVehicleTime')
ModeToSchool = Variable('ModeToSchool')
ReportedDuration = Variable('ReportedDuration')
CoderegionCAR = Variable('CoderegionCAR')
age = Variable('age')
age_category = Variable('age_category')
normalized_weight = Variable('normalized_weight')

ScaledIncome = Variable('ScaledIncome')
age_65_more = Variable('age_65_more')
moreThanOneCar = Variable('moreThanOneCar')
moreThanOneBike = Variable('moreThanOneBike')
individualHouse = Variable('individualHouse')
male = Variable('male')
haveChildren = Variable('haveChildren')
haveGA = Variable('haveGA')
highEducation = Variable('highEducation')
childCenter = Variable('childCenter')
childSuburb = Variable('childSuburb')
TimePT_scaled = Variable('TimePT_scaled')
TimePT_hour = Variable('TimePT_hour')
TimeCar_scaled = Variable('TimeCar_scaled')
TimeCar_hour = Variable('TimeCar_hour')
MarginalCostPT_scaled = Variable('MarginalCostPT_scaled')
CostCarCHF_scaled = Variable('CostCarCHF_scaled')
distance_km_scaled = Variable('distance_km_scaled')
PurpHWH = Variable('PurpHWH')
PurpOther = Variable('PurpOther')
livesInUrbanArea = Variable('livesInUrbanArea')
household_size = Variable('household_size')
income_category = Variable('income_category')
occupation_status = Variable('occupation_status')
