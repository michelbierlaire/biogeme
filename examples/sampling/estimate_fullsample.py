"""File estimate.py. Estimates the models with the full sample

:author: Michel Bierlaire, EPFL
:date: Mon Jan  9 13:24:51 2023

"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable

N_ALT = 100
FILE_ALT = 'restaurants.dat'
DATA_FILE = 'fullsample/obs_fullsample.dat'

# Read the data
df = pd.read_csv(DATA_FILE)
print('Input: ', DATA_FILE)
database = db.Database('restaurant', df)

beta_rating = Beta('beta_rating', 0, None, None, 0)
beta_price = Beta('beta_price', 0, None, None, 0)
beta_chinese = Beta('beta_chinese', 0, None, None, 0)
beta_japanese = Beta('beta_japanese', 0, None, None, 0)
beta_korean = Beta('beta_korean', 0, None, None, 0)
beta_indian = Beta('beta_indian', 0, None, None, 0)
beta_french = Beta('beta_french', 0, None, None, 0)
beta_mexican = Beta('beta_mexican', 0, None, None, 0)
beta_lebanese = Beta('beta_lebanese', 0, None, None, 0)
beta_ethiopian = Beta('beta_ethiopian', 0, None, None, 0)
beta_log_dist = Beta('beta_log_dist', 0, None, None, 0)

mu_nested = Beta('mu_nested', 1, 1, None, 0)
mu_asian = Beta('mu_asian', 1, 1, None, 0)
mu_downtown = Beta('mu_downtown', 1, 1, None, 0)

V = {
    i: (
        beta_rating * Variable(f'rating_{i}')
        + beta_price * Variable(f'price_{i}')
        + beta_chinese * Variable(f'category_Chinese_{i}')
        + beta_japanese * Variable(f'category_Japanese_{i}')
        + beta_korean * Variable(f'category_Korean_{i}')
        + beta_indian * Variable(f'category_Indian_{i}')
        + beta_french * Variable(f'category_French_{i}')
        + beta_mexican * Variable(f'category_Mexican_{i}')
        + beta_lebanese * Variable(f'category_Lebanese_{i}')
        + beta_ethiopian * Variable(f'category_Ethiopian_{i}')
        + beta_log_dist * Variable(f'log_dist_{i}')
    )
    for i in range(N_ALT)
}

logprob_logit = models.loglogit(V, None, Variable('choice_logit'))
biogeme_logit = bio.BIOGEME(database, logprob_logit)
biogeme_logit.modelName = 'restaurant_logit'
biogeme_logit.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})
print('Estimating logit model')
results_logit = biogeme_logit.estimate(recycle=True)
print('Logit model')
print(results_logit.getEstimatedParameters())

alternatives = pd.read_csv(FILE_ALT)
mask = alternatives['Asian'] == 1
asian_ids = set(alternatives[mask]['ID'])
others = set(list(alternatives['ID'])).difference(asian_ids)

the_nest = ((mu_nested, asian_ids),)
other_nests = tuple((1.0, [i]) for i in others)
all_nests_nested = (the_nest) + other_nests

logprob_nested = models.lognested(V, None, all_nests_nested, Variable('choice_nested'))
biogeme_nested = bio.BIOGEME(database, logprob_nested)
biogeme_nested.modelName = 'restaurant_nested'
biogeme_nested.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})
print('Estimating nested model')
results_nested = biogeme_nested.estimate(recycle=True)
print('Nested model')
print(results_nested.getEstimatedParameters())

mask_downtown = alternatives['downtown'] == 1
downtown_ids = set(alternatives[mask_downtown]['ID'])


both_ids = asian_ids & downtown_ids
only_asian = asian_ids - both_ids
only_downtown = downtown_ids - both_ids
other_ids = set(range(N_ALT)) - asian_ids - downtown_ids

alpha_both = {k: 0.5 for k in both_ids}
alpha_asian = {k: 1 for k in only_asian}
alpha_downtown = {k: 1 for k in only_downtown}

alpha_asian = {**alpha_asian, **alpha_both}
alpha_downtown = {**alpha_downtown, **alpha_both}


nest_asian = (mu_asian, alpha_asian)
nest_downtown = (mu_downtown, alpha_downtown)
other_nests = ((1, {k: 1}) for k in other_ids)
all_nests_cnl = (nest_asian, nest_downtown) + tuple(other_nests)

logprob_cnl = models.logcnl(V, None, all_nests_cnl, Variable('choice_cnl'))
biogeme_cnl = bio.BIOGEME(database, logprob_cnl)
biogeme_cnl.modelName = 'restaurant_cnl'
biogeme_cnl.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})
print('Estimating CNL model')
results_cnl = biogeme_cnl.estimate(recycle=True)
print('CNL model')
print(results_cnl.getEstimatedParameters())
