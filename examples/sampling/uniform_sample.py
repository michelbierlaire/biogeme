"""Sample alternatives

:author: Michel Bierlaire
:date: Sat Apr 15 16:01:41 2023

Generate a file with with choice data using:

    - a file containing the description of all alternatives,
          associated with an ID,
    - a file containing observations of individuals, associated with
          socio-economic characteristics, and an observed choice.

The choice must be a valid ID in the file of alternatives.  In the
choice data, a set of unchosen alternatives is sampled for each
individual observation using importance sampling. The correction term
for the model estimation is calculated and introduced in the file as
well.

In the generated choice data, alternative 0 is always the chosen one.
The name of the attributes of the alternatives are the names from the
original file containnig all alternatives, appended with the ID of the
alternative in the sample.

For instance, if there are 3 alternatives sampled per individual, and
the original fime contains the attributed "price", the choice data
will contain the columns "price_0", "price_1" and "price_2".

"""

import numpy as np
import pandas as pd
from biogeme.sampling import StratumTuple, sampling_of_alternatives

SAMPLE_SIZE = 10
FILE_NAME = 'uniformly_sampled_alternatives.dat'

alternatives = pd.read_csv('restaurants.dat')
ID_COLUMN = 'ID'

observations = pd.read_csv('obs_choice.dat')
CHOICE_COLUMN = 'choice'

# Set with all restaurants
all_alternatives = set(list(alternatives[ID_COLUMN]))

partition = (StratumTuple(subset=all_alternatives, sample_size=SAMPLE_SIZE),)

# If the script is executed, the alternatives are actually sampled. If
# it is imported into another script, nothing is actually done.

if __name__ == '__main__':
    sample = sampling_of_alternatives(
        partition=partition,
        individuals=observations,
        choice_column=CHOICE_COLUMN,
        alternatives=alternatives,
        id_column=ID_COLUMN,
        always_include_chosen=True,
    )

    sample.to_csv(
        FILE_NAME,
        index=False,
    )

    print(f'File {FILE_NAME} has been generated')
