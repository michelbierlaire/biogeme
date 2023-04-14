""" Estimate the parameters of the logit models with the generated samples

:author: Michel Bierlaire
:date: Mon Jan  9 08:08:56 2023
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log

from constants import N_ALT, INSTANCES, partitions

MODEL = 'logit'


def estimate(the_database, the_model_name):
    """Estimate the parameters for a given database

    :param the_database: database to be used for estimation
    :type the_database: biogeme.database.Database

    :param the_model_name: name to be used for the output file
    :type the_model_name: str
    """

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
    logprob = models.loglogit_sampling(V, None, log_probability, 0)

    # Create the Biogeme object
    biogeme = bio.BIOGEME(the_database, logprob)
    biogeme.modelName = the_model_name

    # Calculate the null log likelihood for reporting.
    biogeme.calculateNullLoglikelihood({i: 1 for i in range(N_ALT)})

    # Estimate the parameters
    results = biogeme.estimate(recycle=True)
    return results


for p in partitions:
    for n in range(INSTANCES):
        # Read the data
        filename = f'samples/sample_{p}_{MODEL}_{N_ALT}_{n}.dat'
        model_name = f'restaurants_{p}_{MODEL}_{N_ALT}_{n}'
        df = pd.read_csv(filename)
        database = db.Database(f'restaurant_{p}_{n}', df)
        # Calculate distsnces
        for i in range(N_ALT):
            database.DefineVariable(
                f'log_dist_{i}',
                log(
                    (
                        (Variable('user_lat') - Variable(f'rest_lat_{i}')) ** 2
                        + (Variable('user_lon') - Variable(f'rest_lon_{i}')) ** 2
                    )
                    ** 0.5
                ),
            )
        estimate(database, model_name)
