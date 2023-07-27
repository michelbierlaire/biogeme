"""File airline_data.py

:author: Michel Bierlaire, EPFL
:date: Fri Mar 31 10:42:52 2023

Read data for the Airline example
"""

# Too constraining
# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable

# Read the data
df = pd.read_csv('airline.dat', sep='\t')

# Update some data
df.loc[df['q17_Gender'] == 99, 'q17_Gender'] = -1
df.loc[df['q20_Education'] == 99, 'q20_Education'] = -1

# High education
df.loc[df['q20_Education'] <= 10, 'education'] = 1
# Medium education
df.loc[df['q20_Education'] <= 5, 'education'] = 2
# Low education
df.loc[df['q20_Education'] <= 2, 'education'] = 3
df.loc[df['q20_Education'] == 99, 'education'] = -1
df.loc[df['q20_Education'] == -1, 'education'] = -1

df.loc[
    df['q11_DepartureOrArrivalIsImportant'] == -1, 'q11_DepartureOrArrivalIsImportant'
] = 0

database = db.Database('airline', df)

ArrivalTimeHours_1 = Variable('ArrivalTimeHours_1')
BestAlternative_1 = Variable('BestAlternative_1')
BestAlternative_2 = Variable('BestAlternative_2')
BestAlternative_3 = Variable('BestAlternative_3')
q11_DepartureOrArrivalIsImportant = Variable('q11_DepartureOrArrivalIsImportant')
q11_DepartureOrArrivalIsImportant = Variable('q11_DepartureOrArrivalIsImportant')
q12_IdealDepTime = Variable('q12_IdealDepTime')
q13_IdealArrTime = Variable('q13_IdealArrTime')
DepartureTimeMins_1 = Variable('DepartureTimeMins_1')
ArrivalTimeMins_1 = Variable('ArrivalTimeMins_1')
DepartureTimeMins_2 = Variable('DepartureTimeMins_2')
ArrivalTimeMins_2 = Variable('ArrivalTimeMins_2')
DepartureTimeMins_3 = Variable('DepartureTimeMins_3')
ArrivalTimeMins_3 = Variable('ArrivalTimeMins_3')
Fare_1 = Variable('Fare_1')
Fare_2 = Variable('Fare_2')
Fare_3 = Variable('Fare_3')
Legroom_1 = Variable('Legroom_1')
Legroom_2 = Variable('Legroom_2')
Legroom_3 = Variable('Legroom_3')
TripTimeHours_1 = Variable('TripTimeHours_1')
TripTimeHours_2 = Variable('TripTimeHours_2')
TripTimeHours_3 = Variable('TripTimeHours_3')
q02_TripPurpose = Variable('q02_TripPurpose')
q03_WhoPays = Variable('q03_WhoPays')
q17_Gender = Variable('q17_Gender')
q20_Education = Variable('q20_Education')
education = Variable('education')


exclude = ArrivalTimeHours_1 == -1
database.remove(exclude)

chosenAlternative = (
    (BestAlternative_1 * 1) + (BestAlternative_2 * 2) + (BestAlternative_3 * 3)
)

DepartureTimeSensitive = database.DefineVariable(
    'DepartureTimeSensitive', q11_DepartureOrArrivalIsImportant == 1
)
ArrivalTimeSensitive = database.DefineVariable(
    'ArrivalTimeSensitive', q11_DepartureOrArrivalIsImportant == 2
)

DesiredDepartureTime = database.DefineVariable('DesiredDepartureTime', q12_IdealDepTime)
DesiredArrivalTime = database.DefineVariable('DesiredArrivalTime', q13_IdealArrTime)
ScheduledDelay_1 = database.DefineVariable(
    'ScheduledDelay_1',
    (DepartureTimeSensitive * (DepartureTimeMins_1 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_1 - DesiredArrivalTime)),
)

ScheduledDelay_2 = database.DefineVariable(
    'ScheduledDelay_2',
    (DepartureTimeSensitive * (DepartureTimeMins_2 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_2 - DesiredArrivalTime)),
)

ScheduledDelay_3 = database.DefineVariable(
    'ScheduledDelay_3',
    (DepartureTimeSensitive * (DepartureTimeMins_3 - DesiredDepartureTime))
    + (ArrivalTimeSensitive * (ArrivalTimeMins_3 - DesiredArrivalTime)),
)

Opt1_SchedDelayEarly = database.DefineVariable(
    'Opt1_SchedDelayEarly', (-(ScheduledDelay_1) * (ScheduledDelay_1 < 0)) / 60
)
Opt2_SchedDelayEarly = database.DefineVariable(
    'Opt2_SchedDelayEarly', (-(ScheduledDelay_2) * (ScheduledDelay_2 < 0)) / 60
)
Opt3_SchedDelayEarly = database.DefineVariable(
    'Opt3_SchedDelayEarly', (-(ScheduledDelay_3) * (ScheduledDelay_3 < 0)) / 60
)

Opt1_SchedDelayLate = database.DefineVariable(
    'Opt1_SchedDelayLate', (ScheduledDelay_1 * (ScheduledDelay_1 > 0)) / 60
)
Opt2_SchedDelayLate = database.DefineVariable(
    'Opt2_SchedDelayLate', (ScheduledDelay_2 * (ScheduledDelay_2 > 0)) / 60
)
Opt3_SchedDelayLate = database.DefineVariable(
    'Opt3_SchedDelayLate', (ScheduledDelay_3 * (ScheduledDelay_3 > 0)) / 60
)
