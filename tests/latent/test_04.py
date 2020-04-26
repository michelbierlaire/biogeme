import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable, RandomVariable, Integrate, log
import unittest

pandas = pd.read_csv("optima.dat",sep='\t')
database = db.Database("optima",pandas)

globals().update(database.variables)

exclude = (Choice == -1.0)
database.remove(exclude)




### Variables

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)
formulaIncome = models.piecewiseFormula(ScaledIncome,[None,4,6,8,10,None],[0.08954209471304636,-0.2209233080453265,0.2591889240542216,-0.5227805784067027,0.08430692986645968])


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

coef_age_65_more = 0.07181055487226333
coef_haveChildren = -0.03758785262127424
coef_haveGA = -0.5785488899700475
coef_highEducation = -0.24726576867313482
coef_individualHouse = -0.08887159771570047
coef_intercept = 0.40149819890908217
coef_male = 0.0661412838697794
coef_moreThanOneBike = -0.2776091744681671
coef_moreThanOneCar = 0.5335541575826122

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

omega = RandomVariable('omega')
density = dist.normalpdf(omega) 
sigma_s = Beta('sigma_s',0.8625193422179722,None,None,0)


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

ASC_CAR = Beta('ASC_CAR',0.7725067037758291,None,None,0)
ASC_SM = Beta('ASC_SM',1.8865188103480808,None,None,0)
BETA_COST_HWH = Beta('BETA_COST_HWH',-1.7800532700436242,None,None,0)
BETA_COST_OTHER = Beta('BETA_COST_OTHER',-0.8176256998217855,None,None,0)
BETA_DIST = Beta('BETA_DIST',-5.809646562001414,None,None,0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL',-1.6818275468466484,None,None,0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF',-17.694645513468497,None,0,0)
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL',-1.2424575875582378,None,None,0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF',-6.279351989351876,None,None,0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME',-0.02949883465222827,None,None,0)
ASC_PT	 = Beta('ASC_PT',0,-10000,10000,1)

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
prob = Integrate(condprob * density,'omega')
loglike = log(prob)

class test_04(unittest.TestCase):
    def testEstimation(self):
        biogeme  = bio.BIOGEME(database,loglike)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-1092.600088343353,2)
        
if __name__ == '__main__':
    unittest.main()



