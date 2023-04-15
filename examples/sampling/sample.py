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
FILE_NAME = 'sampled_alternatives.dat'

alternatives = pd.read_csv('restaurants.dat')
ID_COLUMN = 'ID'

observations = pd.read_csv('obs_choice.dat')
CHOICE_COLUMN = 'choice'

# Set of Asian restaurants
asian = set(alternatives[alternatives['Asian'] == 1][ID_COLUMN])

# Set of restaurants located in downtown
downtown = set(alternatives[alternatives['downtown'] == 1][ID_COLUMN])

# Set with all restaurants
all_alternatives = set(list(alternatives[ID_COLUMN]))

# Set of Asian restaurants in downtown
asian_and_downtown = asian & downtown

# Set of Asian restaurants, and of restaurants in downtown
asian_or_downtown = asian | downtown

# Set of Asian restaurants not in downtown
only_asian = asian - asian_and_downtown

# Set of non Asian restaurants in downtown
only_downtown = downtown - asian_and_downtown

# Set of restaurants that are neither Asian nor in downtown
others = all_alternatives - asian_or_downtown

half = int(np.floor(SAMPLE_SIZE / 2))
quarter = int(np.floor(SAMPLE_SIZE / 4))

# Importance sample.
# About 25% of the sample are Asian restaurants in downtown
# About 25% of the sample are Asian restaurants not in downtown
# About 25% of the sample are non Asian restaurants in downtown
# About 25% of the sample are meither Asian not in downtown
partition_asian_downtown = (
    StratumTuple(subset=asian_and_downtown, sample_size=quarter),
    StratumTuple(subset=only_asian, sample_size=quarter),
    StratumTuple(subset=only_downtown, sample_size=quarter),
    StratumTuple(subset=others, sample_size=SAMPLE_SIZE - 3 * quarter),
)

# If the script is executed, the alternatives are actually sampled. If
# it is imported into another script, nothing is actually done.

if __name__ == '__main__':
    sample = sampling_of_alternatives(
        partition=partition_asian_downtown,
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
