import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import unittest
from biogeme.expressions import Beta, DefineVariable, log, bioDraws, MonteCarlo, PanelLikelihoodTrajectory


pandas = pd.read_csv("swissmetro.dat",sep='\t')
database = db.Database("swissmetro",pandas)
database.panel("ID")

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

globals().update(database.variables)

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) +  ( CHOICE == 0 )) > 0
database.remove(exclude)



ASC_CAR = Beta('ASC_CAR',0.136,None,None,0)
ASC_TRAIN = Beta('ASC_TRAIN',-1,None,None,0)
ASC_SM = Beta('ASC_SM',0,None,None,1)
B_TIME = Beta('B_TIME',-6.3,None,0,0)
B_COST = Beta('B_COST',-3.29,None,0,0)

SIGMA_CAR = Beta('SIGMA_CAR',3.7,None,None,0)
SIGMA_SM = Beta('SIGMA_SM',0.759,None,None,0)
SIGMA_TRAIN = Beta('SIGMA_TRAIN',3.02,None,None,0)

# Define a random parameter, normally distirbuted, designed to be used
# for Monte-Carlo simulation
EC_CAR = SIGMA_CAR * bioDraws('EC_CAR','NORMAL')
EC_SM = SIGMA_SM * bioDraws('EC_SM','NORMAL')
EC_TRAIN = SIGMA_TRAIN * bioDraws('EC_TRAIN','NORMAL')

SM_COST =  SM_CO   * (  GA   ==  0  ) 
TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)

# For latent class 1, whete the time coefficient is zero
V11 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED  + EC_TRAIN
V12 = ASC_SM + B_COST * SM_COST_SCALED + EC_SM
V13 = ASC_CAR + B_COST * CAR_CO_SCALED + EC_CAR

V1 = {1: V11,
      2: V12,
      3: V13}

# For latent class 2, whete the time coefficient is estimated
V21 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + EC_TRAIN
V22 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + EC_SM
V23 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED + EC_CAR

V2 = {1: V21,
      2: V22,
      3: V23}


# Associate the availability conditions with the alternatives

CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)

av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}


# Class membership model
# Class membership model
CLASS_CTE = Beta('CLASS_CTE',0,None,None,0)
CLASS_INC  = Beta('CLASS_INC',0,None,None,0)
W1 = CLASS_CTE + CLASS_INC * INCOME
probClass1 = models.logit({1:W1,2:0},None,1)
probClass2 = models.logit({1:W1,2:0},None,2)

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 1
prob1 = PanelLikelihoodTrajectory(models.logit(V1,av,CHOICE))

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 2
prob2 = PanelLikelihoodTrajectory(models.logit(V2,av,CHOICE))

# Conditional to the random variables, likelihood for the individual.
probIndiv = probClass1 * prob1 + probClass2 * prob2

# We integrate over the random variables using Monte-Carlo
logprob = log(MonteCarlo(probIndiv))

class test_16(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database,logprob,numberOfDraws=5,seed=10)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-4102.692581020569,2)

    
if __name__ == '__main__':
    unittest.main()


