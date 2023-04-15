""" Model specification for the choice of restaurants

:author: Michel Bierlaire
:date: Sat Apr 15 16:48:40 2023
"""

from biogeme.expressions import Beta, Variable
from sample import SAMPLE_SIZE as N_ALT

# Parameters to estimate
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

# Correction for sampling of alternatives
log_probability = {i: Variable(f'_log_proba_{i}') for i in range(N_ALT)}

# Utility functions. Not that the correction temr is not included in
# the utility functions themselves. It will be included in the call to
# the function loglogit_sampling.

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
