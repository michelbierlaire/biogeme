import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable, RandomVariable, Integrate, log, exp, Numeric, bioNormalCdf, Elem
import unittest

pandas = pd.read_csv("optima.dat",sep='\t')
database = db.Database("optima",pandas)

globals().update(database.variables)

exclude = (Choice == -1.0)
database.remove(exclude)


### Variables

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)
formulaIncome = models.piecewiseFormula(ScaledIncome,[None,4,6,8,10,None],[0.15012511354122204,-0.287364030650875,0.3378758520316341,-0.6808138631421441,0.1191396581172788])

age_65_more = DefineVariable('age_65_more',age >= Numeric(65),database)
moreThanOneCar = DefineVariable('moreThanOneCar',NbCar > 1,database)
moreThanOneBike = DefineVariable('moreThanOneBike',NbBicy > 1,database)
individualHouse = DefineVariable('individualHouse',\
                                 HouseType == 1,database)
male = DefineVariable('male',Gender == 1,database)
haveChildren = DefineVariable('haveChildren',\
                              ((FamilSitu == 3)+(FamilSitu == 4)) > 0,database)
haveGA = DefineVariable('haveGA',GenAbST == 1,database)
highEducation = DefineVariable('highEducation', Education >= 6,database)

### Coefficients
# Read the estimates from the structural equation estimation, and use
# them as starting values


coef_age_65_more = Beta('coef_age_65_more',0.03623576587917444,None,None,0)
coef_haveChildren = Beta('coef_haveChildren',-0.027979246306050195,None,None,0)
coef_haveGA = Beta('coef_haveGA',-0.749193972835575,None,None,0)
coef_highEducation = Beta('coef_highEducation',-0.25961437893330336,None,None,0)
coef_individualHouse = Beta('coef_individualHouse',-0.1162986643952338,None,None,0)
coef_intercept = Beta('coef_intercept',0.3534782074218524,None,None,0)
coef_male = Beta('coef_male',0.07950619870640736,None,None,0)
coef_moreThanOneBike = Beta('coef_moreThanOneBike',-0.36347727873174407,None,None,0)
coef_moreThanOneCar = Beta('coef_moreThanOneCar',0.714788103177542,None,None,0)

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

omega = RandomVariable('omega')
density = dist.normalpdf(omega) 
sigma_s = Beta('sigma_s',0.8617225357645392,None,None,0)

CARLOVERS = \
coef_intercept +\
coef_age_65_more * age_65_more +\
formulaIncome+\
coef_moreThanOneCar * moreThanOneCar +\
coef_moreThanOneBike * moreThanOneBike +\
coef_individualHouse * individualHouse +\
coef_male * male +\
coef_haveChildren * haveChildren +\
coef_haveGA * haveGA +\
coef_highEducation * highEducation +\
sigma_s * omega


### Measurement equations

INTER_Envir01 = Beta('INTER_Envir01',0,None,None,1)
INTER_Envir02 = Beta('INTER_Envir02',0.45955480351421363,None,None,0)
INTER_Envir03 = Beta('INTER_Envir03',-0.36695603516684655,None,None,0)
INTER_Mobil11 = Beta('INTER_Mobil11',0.42001349200736526,None,None,0)
INTER_Mobil14 = Beta('INTER_Mobil14',-0.17303786841995375,None,None,0)
INTER_Mobil16 = Beta('INTER_Mobil16',0.14720885503655662,None,None,0)
INTER_Mobil17 = Beta('INTER_Mobil17',0.13801628954741468,None,None,0)

B_Envir01_F1 = Beta('B_Envir01_F1',-1,None,None,1)
B_Envir02_F1 = Beta('B_Envir02_F1',-0.45602731938035374,None,None,0)
B_Envir03_F1 = Beta('B_Envir03_F1',0.4828530641566324,None,None,0)
B_Mobil11_F1 = Beta('B_Mobil11_F1',0.5699160671155711,None,None,0)
B_Mobil14_F1 = Beta('B_Mobil14_F1',0.5747531819855859,None,None,0)
B_Mobil16_F1 = Beta('B_Mobil16_F1',0.5260309471180279,None,None,0)
B_Mobil17_F1 = Beta('B_Mobil17_F1',0.5189785151350396,None,None,0)


MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01',1,None,None,1)
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02',0.9202761776516004,None,None,0)
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03',0.8584143518703582,None,None,0)
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11',0.8969716757915364,None,None,0)
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14',0.7605962891050919,None,None,0)
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16',0.8732950988286952,None,None,0)
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17',0.8749250793400859,None,None,0)

delta_1 = Beta('delta_1',0.328391151898221,None,None,0)
delta_2 = Beta('delta_2',0.9914737883100323,None,None,0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1 
tau_3 = delta_1
tau_4 = delta_1 + delta_2

Envir01_tau_1 = (tau_1-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_2 = (tau_2-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_3 = (tau_3-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_4 = (tau_4-MODEL_Envir01) / SIGMA_STAR_Envir01
IndEnvir01 = {
    1: bioNormalCdf(Envir01_tau_1),
    2: bioNormalCdf(Envir01_tau_2)-bioNormalCdf(Envir01_tau_1),
    3: bioNormalCdf(Envir01_tau_3)-bioNormalCdf(Envir01_tau_2),
    4: bioNormalCdf(Envir01_tau_4)-bioNormalCdf(Envir01_tau_3),
    5: 1-bioNormalCdf(Envir01_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir01 = Elem(IndEnvir01, Envir01)


Envir02_tau_1 = (tau_1-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_2 = (tau_2-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_3 = (tau_3-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_4 = (tau_4-MODEL_Envir02) / SIGMA_STAR_Envir02
IndEnvir02 = {
    1: bioNormalCdf(Envir02_tau_1),
    2: bioNormalCdf(Envir02_tau_2)-bioNormalCdf(Envir02_tau_1),
    3: bioNormalCdf(Envir02_tau_3)-bioNormalCdf(Envir02_tau_2),
    4: bioNormalCdf(Envir02_tau_4)-bioNormalCdf(Envir02_tau_3),
    5: 1-bioNormalCdf(Envir02_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir02 = Elem(IndEnvir02, Envir02)

Envir03_tau_1 = (tau_1-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_2 = (tau_2-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_3 = (tau_3-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_4 = (tau_4-MODEL_Envir03) / SIGMA_STAR_Envir03
IndEnvir03 = {
    1: bioNormalCdf(Envir03_tau_1),
    2: bioNormalCdf(Envir03_tau_2)-bioNormalCdf(Envir03_tau_1),
    3: bioNormalCdf(Envir03_tau_3)-bioNormalCdf(Envir03_tau_2),
    4: bioNormalCdf(Envir03_tau_4)-bioNormalCdf(Envir03_tau_3),
    5: 1-bioNormalCdf(Envir03_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir03 = Elem(IndEnvir03, Envir03)

Mobil11_tau_1 = (tau_1-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_2 = (tau_2-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_3 = (tau_3-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_4 = (tau_4-MODEL_Mobil11) / SIGMA_STAR_Mobil11
IndMobil11 = {
    1: bioNormalCdf(Mobil11_tau_1),
    2: bioNormalCdf(Mobil11_tau_2)-bioNormalCdf(Mobil11_tau_1),
    3: bioNormalCdf(Mobil11_tau_3)-bioNormalCdf(Mobil11_tau_2),
    4: bioNormalCdf(Mobil11_tau_4)-bioNormalCdf(Mobil11_tau_3),
    5: 1-bioNormalCdf(Mobil11_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil11 = Elem(IndMobil11, Mobil11)

Mobil14_tau_1 = (tau_1-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_2 = (tau_2-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_3 = (tau_3-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_4 = (tau_4-MODEL_Mobil14) / SIGMA_STAR_Mobil14
IndMobil14 = {
    1: bioNormalCdf(Mobil14_tau_1),
    2: bioNormalCdf(Mobil14_tau_2)-bioNormalCdf(Mobil14_tau_1),
    3: bioNormalCdf(Mobil14_tau_3)-bioNormalCdf(Mobil14_tau_2),
    4: bioNormalCdf(Mobil14_tau_4)-bioNormalCdf(Mobil14_tau_3),
    5: 1-bioNormalCdf(Mobil14_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil14 = Elem(IndMobil14, Mobil14)

Mobil16_tau_1 = (tau_1-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_2 = (tau_2-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_3 = (tau_3-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_4 = (tau_4-MODEL_Mobil16) / SIGMA_STAR_Mobil16
IndMobil16 = {
    1: bioNormalCdf(Mobil16_tau_1),
    2: bioNormalCdf(Mobil16_tau_2)-bioNormalCdf(Mobil16_tau_1),
    3: bioNormalCdf(Mobil16_tau_3)-bioNormalCdf(Mobil16_tau_2),
    4: bioNormalCdf(Mobil16_tau_4)-bioNormalCdf(Mobil16_tau_3),
    5: 1-bioNormalCdf(Mobil16_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil16 = Elem(IndMobil16, Mobil16)

Mobil17_tau_1 = (tau_1-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_2 = (tau_2-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_3 = (tau_3-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_4 = (tau_4-MODEL_Mobil17) / SIGMA_STAR_Mobil17
IndMobil17 = {
    1: bioNormalCdf(Mobil17_tau_1),
    2: bioNormalCdf(Mobil17_tau_2)-bioNormalCdf(Mobil17_tau_1),
    3: bioNormalCdf(Mobil17_tau_3)-bioNormalCdf(Mobil17_tau_2),
    4: bioNormalCdf(Mobil17_tau_4)-bioNormalCdf(Mobil17_tau_3),
    5: 1-bioNormalCdf(Mobil17_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil17 = Elem(IndMobil17, Mobil17)

# Choice model
# Read the estimates from the sequential estimation, and use
# them as starting values


ASC_CAR = Beta('ASC_CAR',1.0807583353265193,None,None,0)
ASC_PT	 = Beta('ASC_PT',0,None,None,1)
ASC_SM = Beta('ASC_SM',0.522400536303002,None,None,0)
BETA_COST_HWH = Beta('BETA_COST_HWH',-1.3710784143244534,None,None,0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER',-0.6540470100896351,None,None,0)
BETA_DIST = Beta('BETA_DIST',-1.1010796012186104,None,None,0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL',-1.0594331210198944,None,None,0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF',-4.81908293736362,None,None,0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL',-1.2435830554082432,None,None,0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF',-0.0001,None,None,0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME',-0.04409308504170328,None,None,0)

TimePT_scaled  = DefineVariable('TimePT_scaled', TimePT   /  200 ,database)
TimeCar_scaled  = DefineVariable('TimeCar_scaled', TimeCar   /  200 ,database)
MarginalCostPT_scaled  = \
 DefineVariable('MarginalCostPT_scaled', MarginalCostPT   /  10 ,database)
CostCarCHF_scaled  = \
 DefineVariable('CostCarCHF_scaled', CostCarCHF   /  10 ,database)
distance_km_scaled  = \
 DefineVariable('distance_km_scaled', distance_km   /  5 ,database)
PurpHWH = DefineVariable('PurpHWH', TripPurpose == 1,database)
PurpOther = DefineVariable('PurpOther', TripPurpose != 1,database)


### DEFINITION OF UTILITY FUNCTIONS:

BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_WAITING_TIME * WaitingTimePT + \
     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH  +\
     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = ASC_CAR + \
      BETA_TIME_CAR * TimeCar_scaled + \
      BETA_COST_HWH * CostCarCHF_scaled * PurpHWH  + \
      BETA_COST_OTHER * CostCarCHF_scaled * PurpOther 

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# Associate utility functions with the numbering of alternatives
V = {0: V0,
     1: V1,
     2: V2}

# Associate the availability conditions with the alternatives.
# In this example all alternatives are available for each individual.
av = {0: 1,
      1: 1,
      2: 1}

# The choice model is a logit, conditional to the value of the latent variable
condprob = models.logit(V,av,Choice)
condlike = P_Envir01 * \
          P_Envir02 * \
          P_Envir03 * \
          P_Mobil11 * \
          P_Mobil14 * \
          P_Mobil16 * \
          P_Mobil17 * \
          condprob

loglike = log(Integrate(condlike * density,'omega'))

class test_04(unittest.TestCase):
    def testEstimation(self):
        biogeme  = bio.BIOGEME(database,loglike)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-18383.069853455232,2)
        
if __name__ == '__main__':
    unittest.main()

