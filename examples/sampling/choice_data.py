"""Processing of the choice data with sampling of alternatives

:author: Michel Bierlaire
:date: Sat Apr 15 16:46:42 2023

"""
import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable, log

from sample import (
    FILE_NAME,
    SAMPLE_SIZE as N_ALT,
)

df = pd.read_csv(FILE_NAME)
database = db.Database('choice_data', df)

# We need to calculate the distances between each individual and each
# restaurant in the sampled choice set.

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
