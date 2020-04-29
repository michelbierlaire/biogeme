"""File 01logitBis.py

:author: Michel Bierlaire, EPFL
:date: Thu Sep  6 15:14:39 2018

 Logit model
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.optimization as opt
import biogeme.models as models
from biogeme.expressions import Beta, DefineVariable,bioLinearUtility

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
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)

# We explicitly mention that the utility functions are linear.
# They are specified as a list of terms.
terms1 = [(B_TIME,TRAIN_TT_SCALED),(B_COST,TRAIN_COST_SCALED)] 
V1 = ASC_TRAIN + bioLinearUtility(terms1)

terms2 = [(B_TIME,SM_TT_SCALED),(B_COST,SM_COST_SCALED)] 
V2 = ASC_SM + bioLinearUtility(terms2)

terms3 = [(B_TIME,CAR_TT_SCALED),(B_COST,CAR_CO_SCALED)] 
V3 = ASC_CAR + bioLinearUtility(terms3)

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
biogeme.modelName = "01logitBis"

# Avoids the generation of the pickle file.  Note: this is included
# only in this example to illustrate the syntax, and not recommended.
# The pickle file contains the results of the estimation and allows
# you to analyze the results later on without re-estimating the model.
biogeme.generatePickle = False

# Estimate the parameters. 
results = biogeme.estimate(algorithm=opt.newtonTrustRegionForBiogeme,bootstrap=100)

# Print the estimated values
betas = results.getBetaValues()
for k,v in betas.items():
    print(f"{k:10}=\t{v:.3g}")

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
