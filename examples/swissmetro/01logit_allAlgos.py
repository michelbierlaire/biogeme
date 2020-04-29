""" File 01logit_allAlgos.py

:author: Michel Bierlaire, EPFL
:date: Sat Sep  7 17:57:16 2019

 Logit model
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.optimization as opt
import biogeme.hamabs as hamabs
import biogeme.models as models
from biogeme.expressions import Beta, DefineVariable

# Read the data
df = pd.read_csv("swissmetro.dat",'\t')
database = db.Database("swissmetro",df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) +  ( CHOICE == 0 )) > 0
database.remove(exclude)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR',0,None,None,0)
ASC_TRAIN = Beta('ASC_TRAIN',0,None,None,0)
ASC_SM = Beta('ASC_SM',0,None,None,1)
B_TIME = Beta('B_TIME',0,None,None,0)
B_COST = Beta('B_COST',0,None,None,0)

# Definition of new variables
SM_COST =  SM_CO   * (  GA   ==  0  ) 
TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

# Definition of new variables: adding columns to the database 
CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)

# Definition of new variables: adding columns to the database 
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)

# Definition of the utility functions
V1 = ASC_TRAIN + \
     B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V,av,CHOICE)

# Define level of verbosity
import biogeme.messaging as msg
logger = msg.bioMessage()
logger.setSilent()
#logger.setWarning()
#logger.setGeneral()
#logger.setDetailed()
        
# Create the Biogeme object
biogeme  = bio.BIOGEME(database,logprob)

algos={'CFSQP':None,
       'scipy':opt.scipy,
       'Line search':opt.newtonLineSearchForBiogeme,
       'Trust region (dogleg)':opt.newtonTrustRegionForBiogeme,
       'Trust region (cg)':opt.newtonTrustRegionForBiogeme,
       'LS-BFGS':opt.bfgsLineSearchForBiogeme,
       'TR-BFGS':opt.bfgsTrustRegionForBiogeme,
       'Simple bounds':opt.simpleBoundsNewtonAlgorithmForBiogeme,
       'HAMABS':hamabs.hamabs}

algoParameters ={'Trust region (dogleg)':{'dogleg':True},
                 'Trust region (cg)':{'dogleg':False}}

results = {}
print("Algorithm             loglike         normg    time          feval diagnostic")
print("+++++++++             +++++++         +++++    ++++          +++++ ++++++++++")

for name,algo in algos.items():
    print(f'Running {name}')
    biogeme.modelName = f"01logit_allAlgos_{name}"
    p = algoParameters.get(name)
    results[name] = biogeme.estimate(algorithm=algo,algoParameters=p)
    print(results[name].data.optimizationMessages)
    g = results[name].data.g


