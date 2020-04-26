import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import unittest
from biogeme.expressions import Beta, DefineVariable, RandomVariable, Integrate, exp, log

pandas = pd.read_csv("optima.dat",sep='\t')
database = db.Database("optima",pandas)

globals().update(database.variables)

exclude = (Choice == -1.0)
database.remove(exclude)




### Variables

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)
formulaIncome = models.piecewiseFormula(ScaledIncome,[None,4,6,8,10,None],[-0.33669944071374086,0.2177481824894475,-0.6224104823788724,1.1592372281254595,-0.35530173883216076])

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

ASC_CAR = Beta('ASC_CAR', 0.4118057751313065,None,None,0)
ASC_SM = Beta('ASC_SM', 1.0134144912346523,None,None,0)
BETA_COST_HWH = Beta('BETA_COST_HWH', -1.7754249429871691,None,None,0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER', -1.5427606901901116,None,None,0)
BETA_DIST = Beta('BETA_DIST', -4.952750818319787,None,None,0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL', -0.1412781305192878,None,None,0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF', -26.647254885078237,None,None,0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL', -0.5048865862522101,None,None,0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF', -4.883762313381118,None,None,0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME', -0.052434936015492456,None,None,0)
coef_age_65_more = Beta('coef_age_65_more', 1.1187240773542817,None,None,0)
coef_haveChildren = Beta('coef_haveChildren', 0.18090193116052988,None,None,0)
coef_haveGA = Beta('coef_haveGA', 4.657846388649241,None,None,0)
coef_highEducation = Beta('coef_highEducation', -1.4305949806468112,None,None,0)
coef_individualHouse = Beta('coef_individualHouse', 0.1408510971738762,None,None,0)
coef_male = Beta('coef_male', -0.6817893725153135,None,None,0)
coef_moreThanOneBike = Beta('coef_moreThanOneBike', 0.3944574007669639,None,None,0)
coef_moreThanOneCar = Beta('coef_moreThanOneCar', -2.2429029134722964,None,None,0)
sigma_s = Beta('sigma_s', 3.580536793824548,None,None,0)

coef_intercept = Beta('coef_intercept',0.0,None,None,1)

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

omega = RandomVariable('omega')
density = dist.normalpdf(omega) 

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

# Choice model


ASC_PT	 = Beta('ASC_PT',0.0,None,None,1)

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

BETA_TIME_PT = BETA_TIME_PT_REF * \
               exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_WAITING_TIME * WaitingTimePT + \
     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH  +\
     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

BETA_TIME_CAR = BETA_TIME_CAR_REF * \
                exp(BETA_TIME_CAR_CL * CARLOVERS)

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
# In this example all alternatives are available 
# for each individual.
av = {0: 1,
      1: 1,
      2: 1}

# The choice model is a logit, conditional to 
# the value of the latent variable
condprob = models.logit(V,av,Choice)
prob = Integrate(condprob * density,'omega')
loglike = log(prob)

class test_03(unittest.TestCase):
    def testEstimation(self):
        biogeme  = bio.BIOGEME(database,loglike)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-1076.6955020804955,2)
        
if __name__ == '__main__':
    unittest.main()


