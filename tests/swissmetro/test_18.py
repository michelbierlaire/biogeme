import pandas as pd
import sys
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.distributions as dist
import unittest
from biogeme.expressions import Beta, DefineVariable, log, Elem


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


B_TIME = Beta('B_TIME',0,None,None,0)
B_COST = Beta('B_COST',0,None,None,0)

tau1	 = Beta('tau1',-1,None,0,0)
delta2	 = Beta('delta2',2,0,None,0)

tau2 = tau1 + delta2


TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)

#  Utility

U = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED 


ChoiceProba = {
    1: 1-dist.logisticcdf(U-tau1),
    2: dist.logisticcdf(U-tau1)- dist.logisticcdf(U-tau2),
    3: dist.logisticcdf(U-tau2) }

logprob = log(Elem(ChoiceProba,CHOICE))

class test_18(unittest.TestCase):
    def testEstimation(self):
        biogeme  = bio.BIOGEME(database,logprob)
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike,-5789.309,2)
    
if __name__ == '__main__':
    unittest.main()


