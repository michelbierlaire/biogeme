import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import unittest
from biogeme.expressions import Beta, DefineVariable

pandas = pd.read_csv("swissmetro.dat",sep='\t')
database = db.Database("swissmetro",pandas)

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



ASC_CAR = Beta('ASC_CAR',0,None,None,0)
ASC_TRAIN = Beta('ASC_TRAIN',0,None,None,0)
ASC_SM = Beta('ASC_SM',0,None,None,1)
B_TIME = Beta('B_TIME',0,None,None,0)
B_COST = Beta('B_COST',0,None,None,0)

MU = Beta('MU',0.5,0.0,1.0,0)



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

V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}


# Associate the availability conditions with the alternatives
CAR_AV_SP = DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)

av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}

#Definition of nests:
# 1: nests parameter
# 2: list of alternatives
existing = 1.0 , [1,3]
future = 1.0 , [2]
nests = existing,future

# The choice model is a nested logit, with availability conditions
logprob = models.lognestedMevMu(V,av,nests,CHOICE,MU)

class test_10(unittest.TestCase):
    def testEstimation(self):
        biogeme  = bio.BIOGEME(database,logprob)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-5236.9,2)
    
if __name__ == '__main__':
    unittest.main()


