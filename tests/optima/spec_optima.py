"""File spec_optima.py

:author: Michel Bierlaire, EPFL
:date: Fri Jul 29 16:01:42 2022

"""
from biogeme import models
from biogeme.expressions import Beta, Variable
import biogeme.segmentation as seg

# Definition of the variables in the data file
Id = Variable('Id')
MarginalCostPT = Variable('MarginalCostPT')
WaitingTimePT = Variable('WaitingTimePT')
CostCarCHF = Variable('CostCarCHF')
NbTransf = Variable('NbTransf')
distance_km = Variable('distance_km')
TimePT = Variable('TimePT')
TimeCar = Variable('TimeCar')
OccupStat = Variable('OccupStat')
LangCode = Variable('LangCode')
CarAvail = Variable('CarAvail')
Education = Variable('Education')
TripPurpose = Variable('TripPurpose')
Prob0 = Variable('Prob0')
Prob1 = Variable('Prob1')
Prob2 = Variable('Prob2')
Choice = Variable('Choice')

TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5

car_avail_segmentation = seg.DiscreteSegmentationTuple(
    variable=CarAvail,
    mapping={1: 'car_avail', 3: 'car_unavail'},
    reference=None,
)

language_segmentation = seg.DiscreteSegmentationTuple(
    variable=LangCode, mapping={1: 'french', 2: 'german'}, reference=None
)

occup_segmentation = seg.DiscreteSegmentationTuple(
    variable=OccupStat, mapping={1: 'full_time', 2: 'part_time', 3: 'others'}, reference=None
)

purpose_segmentation = seg.DiscreteSegmentationTuple(
    variable=TripPurpose, mapping={1: 'work', 2: 'not_work'}, reference=None
)

education_segmentation = seg.DiscreteSegmentationTuple(
    variable=Education,
    mapping={
        3: 'vocational',
        4: 'high_school',
        6: 'higher_education',
        7: 'university',
    },
    reference=None
)

ASC_PT_base = Beta('ASC_PT', 0, None, None, 0)

ASC_PT = (
    seg.Segmentation(
        ASC_PT_base,
        [car_avail_segmentation, language_segmentation],
    ).segmented_beta()
)

ASC_CAR_base = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR = (
    seg.Segmentation(
        ASC_CAR_base,
        [car_avail_segmentation, language_segmentation],
    ).segmented_beta()
)

BETA_TIME_base = Beta('BETA_TIME', 0, None, None, 0)
BETA_TIME = (
    seg.Segmentation(
        BETA_TIME_base,
        [occup_segmentation]
    ).segmented_beta()
)

BETA_COST_PT = Beta('BETA_COST_PT', 0, None, None, 0)
BETA_COST_CAR = Beta('BETA_COST_CAR', 0, None, None, 0)

BETA_WAITING_base = Beta('BETA_WAITING', 0, None, None, 0)
BETA_WAITING = seg.Segmentation(BETA_WAITING_base, [purpose_segmentation]).segmented_beta()

BETA_DIST_base = Beta('BETA_DIST', 0, None, None, 0)
BETA_DIST = seg.Segmentation(BETA_DIST_base, [education_segmentation]).segmented_beta()

LAMBDA_COST = 0.3214999879822265

V_PT = (
    ASC_PT
    + BETA_TIME * TimePT_scaled
    + BETA_COST_PT * models.boxcox(MarginalCostPT_scaled, LAMBDA_COST)
    + BETA_WAITING * WaitingTimePT**0.5
)

V_CAR = (
    ASC_CAR
    + BETA_TIME * TimeCar_scaled
    + BETA_COST_CAR * models.boxcox(CostCarCHF_scaled, LAMBDA_COST)
)

V_SM = BETA_DIST * distance_km_scaled


V = {0: V_PT, 1: V_CAR, 2: V_SM}

av = {0: 1, 1: CarAvail != 3, 2: 1}
