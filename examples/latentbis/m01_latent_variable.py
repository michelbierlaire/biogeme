""" File m01_latent_variable.py

Measurement equation where the indicators are discrete.
Ordered probit.

:author: Michel Bierlaire, EPFL
:date: Tue Jul  6 15:47:26 2021

"""
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio

from biogeme.expressions import (
    Beta,
    Variable,
    log,
    Elem,
    bioNormalCdf,
    Variable,
    bioMultSum,
)


# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# Extract the variables fromm the data file
Choice = Variable('Choice')
Gender = Variable('Gender')
FamilSitu = Variable('FamilSitu')
Education = Variable('Education')
ResidChild = Variable('ResidChild')
SocioProfCat = Variable('SocioProfCat')
age = Variable('age')

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Definition of other variables
male = database.DefineVariable('male', Gender == 1)

haveChildren = database.DefineVariable(
    'haveChildren', ((FamilSitu == 3) + (FamilSitu == 4)) > 0
)

highEducation = database.DefineVariable('highEducation', Education >= 6)

childCenter = database.DefineVariable(
    'childCenter', ((ResidChild == 1) + (ResidChild == 2)) > 0
)

childSuburb = database.DefineVariable(
    'childSuburb', ((ResidChild == 3) + (ResidChild == 4)) > 0
)

# Parameters for the structural equation
coef_intercept = Beta('coef_intercept', 0.0, None, None, 0)
coef_age_30_less = Beta('coef_age_30_less', 0.0, None, None, 0)
coef_male = Beta('coef_male', 0.0, None, None, 0)
coef_haveChildren = Beta('coef_haveChildren', 0.0, None, None, 0)
coef_highEducation = Beta('coef_highEducation', 0.0, None, None, 0)
coef_artisans = Beta('coef_artisans', 0.0, None, None, 0)
coef_employees = Beta('coef_employees', 0.0, None, None, 0)
coef_child_center = Beta('coef_child_center', 0.0, None, None, 0)
coef_child_suburb = Beta('coef_child_suburb', 0.0, None, None, 0)

# Latent variable: structural equation
ACTIVELIFE = (
    coef_intercept
    + coef_child_center * childCenter
    + coef_child_suburb * childSuburb
    + coef_highEducation * highEducation
    + coef_artisans * (SocioProfCat == 5)
    + coef_employees * (SocioProfCat == 6)
    + coef_age_30_less * (age <= 30)
    + coef_male * male
    + coef_haveChildren * haveChildren
)

### Measurement equations

indicators = [
    'ResidCh01',
    'ResidCh04',
    'ResidCh05',
    'ResidCh06',
    'LifSty07',
    'LifSty10',
]

# We define the intercept parameters. The first one is normalized to 0.
INTER = {k: Beta(f'INTER_{k}', 0, None, None, 0) for k in indicators[1:]}
INTER[indicators[0]] = Beta(f'INTER_{indicators[0]}', 0, None, None, 1)

# We define the coefficients. The first one is normalized to 1.
B = {k: Beta(f'B_{k}', 0, None, None, 0) for k in indicators[1:]}
B[indicators[0]] = Beta(f'B_{indicators[0]}', 1, None, None, 1)

# We define the measurement equations for each indicator
MODEL = {k: INTER[k] + B[k] * ACTIVELIFE for k in indicators}

# We define the scale parameters of the error terms.
SIGMA_STAR = {
    k: Beta(f'SIGMA_STAR_{k}', 1, 1.0e-5, None, 0) for k in indicators[1:]
}
SIGMA_STAR[indicators[0]] = Beta(
    f'SIGMA_STAR_{indicators[0]}', 1, None, None, 1
)

delta_1 = Beta('delta_1', 0.1, 1.0e-5, None, 0)
delta_2 = Beta('delta_2', 0.2, 1.0e-5, None, 0)
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1
tau_3 = delta_1
tau_4 = delta_1 + delta_2

tau_1_residual = {k: (tau_1 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_2_residual = {k: (tau_2 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_3_residual = {k: (tau_3 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
tau_4_residual = {k: (tau_4 - MODEL[k]) / SIGMA_STAR[k] for k in indicators}
Ind = {
    k: {
        1: bioNormalCdf(tau_1_residual[k]),
        2: bioNormalCdf(tau_2_residual[k]) - bioNormalCdf(tau_1_residual[k]),
        3: bioNormalCdf(tau_3_residual[k]) - bioNormalCdf(tau_2_residual[k]),
        4: bioNormalCdf(tau_4_residual[k]) - bioNormalCdf(tau_3_residual[k]),
        5: 1 - bioNormalCdf(tau_4_residual[k]),
        6: 1.0,
        -1: 1.0,
        -2: 1.0,
    }
    for k in indicators
}

logP = {k: log(Elem(Ind[k], Variable(k))) for k in indicators}
loglike = bioMultSum(logP)

# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike)
biogeme.modelName = 'm01_latent_variable'

# Estimate the parameters
results = biogeme.estimate()

print(results.getEstimatedParameters())
