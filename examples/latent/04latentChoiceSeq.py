"""File 04latentChoiceSeq.py

Choice model with the latent variable.
Mixture of logit.
Measurement equation for the indicators.
Sequential estimation.

:author: Michel Bierlaire, EPFL
:date: Tue Sep 10 08:13:18 2019

"""
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.distributions as dist
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable, RandomVariable, exp, log, Integrate

# Read the data
df = pd.read_csv("optima.dat",sep='\t')
database = db.Database("optima",df)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
exclude = (Choice == -1.0)
database.remove(exclude)


### Variables

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)
thresholds = [4,6,8,10]
ContIncome = models.piecewise(ScaledIncome,thresholds)
ContIncome_0_4000 = ContIncome[0]
ContIncome_4000_6000 = ContIncome[1]
ContIncome_6000_8000 = ContIncome[2]
ContIncome_8000_10000 = ContIncome[3]
ContIncome_10000_more = ContIncome[4]

# Definition of other variables
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
# Read the estimates from the structural equation estimation
structResults = res.bioResults(pickleFile='02oneLatentOrdered.pickle')
structBetas = structResults.getBetaValues()

coef_intercept = structBetas['coef_intercept']
coef_age_65_more = structBetas['coef_age_65_more']
coef_haveGA = structBetas['coef_haveGA']
coef_ContIncome_0_4000 = structBetas['coef_ContIncome_0_4000']
coef_ContIncome_4000_6000 = structBetas['coef_ContIncome_4000_6000']
coef_ContIncome_6000_8000 = structBetas['coef_ContIncome_6000_8000']
coef_ContIncome_8000_10000 = structBetas['coef_ContIncome_8000_10000']
coef_ContIncome_10000_more = structBetas['coef_ContIncome_10000_more']
coef_moreThanOneCar = structBetas['coef_moreThanOneCar']
coef_moreThanOneBike = structBetas['coef_moreThanOneBike']
coef_individualHouse = structBetas['coef_individualHouse']
coef_male = structBetas['coef_male']
coef_haveChildren = structBetas['coef_haveChildren']
coef_highEducation = structBetas['coef_highEducation']

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

# Define a random parameter, normally distributed, designed to be used
# for numerical integration
omega = RandomVariable('omega')
density = dist.normalpdf(omega) 
sigma_s = Beta('sigma_s',1,-1000,1000,0)

CARLOVERS = \
            coef_intercept +\
            coef_age_65_more * age_65_more +\
            coef_ContIncome_0_4000 * ContIncome_0_4000 +\
            coef_ContIncome_4000_6000 * ContIncome_4000_6000 +\
            coef_ContIncome_6000_8000 * ContIncome_6000_8000 +\
            coef_ContIncome_8000_10000 * ContIncome_8000_10000 +\
            coef_ContIncome_10000_more * ContIncome_10000_more +\
            coef_moreThanOneCar * moreThanOneCar +\
            coef_moreThanOneBike * moreThanOneBike +\
            coef_individualHouse * individualHouse +\
            coef_male * male +\
            coef_haveChildren * haveChildren +\
            coef_haveGA * haveGA +\
            coef_highEducation * highEducation +\
            sigma_s * omega


# Choice model


ASC_CAR	 = Beta('ASC_CAR',0,-10000,10000,0)
ASC_PT	 = Beta('ASC_PT',0,-10000,10000,1)
ASC_SM	 = Beta('ASC_SM',0,-10000,10000,0)
BETA_COST_HWH = Beta('BETA_COST_HWH',0.0,-10000,10000,0 )
BETA_COST_OTHER = Beta('BETA_COST_OTHER',0.0,-10000,10000,0 )
BETA_DIST	 = Beta('BETA_DIST',0.0,-10000,10000,0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF',0.0,-10000,0,0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL',0.0,-10,10,0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF',0.0,-10000,0,0 )
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL',0.0,-10,10,0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME',0.0,-10000,10000,0 )

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

### Definition of utility functions:

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

# Conditional to omega, we have a logit model (called the kernel)
condprob = models.logit(V,av,Choice)

# We integrate over omega using numerical integration
loglike = log(Integrate(condprob * density,'omega'))

# Define level of verbosity
import biogeme.messaging as msg
logger = msg.bioMessage()
#logger.setSilent()
#logger.setWarning()
logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "04latentChoiceSeq"

# Estimate the parameters
results = biogeme.estimate(algorithm=None)

print(f"Estimated betas: {len(results.data.betaValues)}")
print(f"Final log likelihood: {results.data.logLike:.3f}")
print(f"Output file: {results.data.htmlFileName}")
results.writeLaTeX()
print(f"LaTeX file: {results.data.latexFileName}")



