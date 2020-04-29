import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
#import biogeme.models as models
import biogeme.loglikelihood as ll
from biogeme.expressions import Beta, DefineVariable, log, Integrate, Elem, bioNormalCdf

pandas = pd.read_csv("optima.dat",sep='\t')
database = db.Database("optima",pandas)

globals().update(database.variables)

exclude = (Choice == -1.0)
database.remove(exclude)



### Variables

ScaledIncome = DefineVariable('ScaledIncome',\
                              CalculatedIncome / 1000,database)

formulaIncome = models.piecewiseFormula(ScaledIncome,[None,4,6,8,10,None],[0.0902761,-0.221283,0.259466,-0.523049,0.084351])

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
coef_intercept = Beta('coef_intercept',0.398165,None,None,0 )
coef_age_65_more = Beta('coef_age_65_more',0.0716533,None,None,0 )
coef_haveGA = Beta('coef_haveGA',-0.578005,None,None,0 )
coef_moreThanOneCar = \
 Beta('coef_moreThanOneCar',0.53301,None,None,0 )
coef_moreThanOneBike = \
 Beta('coef_moreThanOneBike',-0.277122,None,None,0 )
coef_individualHouse = \
 Beta('coef_individualHouse',-0.0885649,None,None,0 )
coef_male = Beta('coef_male',0.0663476,None,None,0 )
coef_haveChildren = Beta('coef_haveChildren',-0.0376042,None,None,0 )
coef_highEducation = Beta('coef_highEducation',-0.246687,None,None,0 )

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

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
coef_highEducation * highEducation


### Measurement equations

INTER_Envir01 = Beta('INTER_Envir01',0,None,None,1)
INTER_Envir02 = Beta('INTER_Envir02',0.348654,None,None,0 )
INTER_Envir03 = Beta('INTER_Envir03',-0.309023,None,None,0 )
INTER_Mobil11 = Beta('INTER_Mobil11',0.337726,None,None,0 )
INTER_Mobil14 = Beta('INTER_Mobil14',-0.130563,None,None,0 )
INTER_Mobil16 = Beta('INTER_Mobil16',0.128293,None,None,0 )
INTER_Mobil17 = Beta('INTER_Mobil17',0.145876,None,None,0 )

B_Envir01_F1 = Beta('B_Envir01_F1',-1,None,None,1)
B_Envir02_F1 = Beta('B_Envir02_F1',-0.431461,None,None,0 )
B_Envir03_F1 = Beta('B_Envir03_F1',0.565903,None,None,0 )
B_Mobil11_F1 = Beta('B_Mobil11_F1',0.483958,None,None,0 )
B_Mobil14_F1 = Beta('B_Mobil14_F1',0.58221,None,None,0 )
B_Mobil16_F1 = Beta('B_Mobil16_F1',0.463139,None,None,0 )
B_Mobil17_F1 = Beta('B_Mobil17_F1',0.368257,None,None,0 )



MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01',1,-10000,10000,1)
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02',0.767063,-10000,10000,0 )
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03',0.717835,-10000,10000,0 )
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11',0.783358,-10000,10000,0 )
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14',0.688264,-10000,10000,0 )
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16',0.754419,-10000,10000,0 )
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17',0.760104,-10000,10000,0 )

delta_1 = Beta('delta_1',0.251983,0,10,0 )
delta_2 = Beta('delta_2',0.759208,0,10,0 )
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


loglike = log(P_Envir01) + \
          log(P_Envir02) + \
          log(P_Envir03) + \
          log(P_Mobil11) + \
          log(P_Mobil14) + \
          log(P_Mobil16) + \
          log(P_Mobil17)


print(database.data.head())
biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "02oneLatentOrdered"
results = biogeme.estimate()
print("Estimated betas: {}".format(len(results.data.betaValues)))

