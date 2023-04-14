""" Sample alternatives

:author: Michel Bierlaire
:date: Sun Jan  8 10:28:03 2023
"""

import os
import numpy as np
import pandas as pd
from biogeme.sampling import StratumTuple, sampling_of_alternatives

SAMPLE_SIZE = 10

models = ('logit', 'nested', 'cnl')

alternatives = pd.read_csv('restaurants.dat')
obs = pd.read_csv('obs_choice.dat')


def generate_samples(name, partition, instances=10):
    """Generates samples for each model based on the partition

    :param name: name of the partition
    :type name: str

    :param partition: definition of the partition
    :type partition: StratumTuple

    :param instances: number of instances of the same sample to be generated
    :type instances: int
    """
    os.makedirs('samples', exist_ok=True)

    for n in range(instances):
        for model in models:
            filename = f'samples/sample_{name}_{model}_{SAMPLE_SIZE}_{n}.dat'
            if os.path.exists(filename):
                print(f'{filename} already exists')
            else:
                print(f'Generate {filename}')
                sample = sampling_of_alternatives(
                    partition=partition,
                    individuals=obs,
                    choice_column=f'choice_{model}',
                    alternatives=alternatives,
                    id_column='ID',
                    always_include_chosen=True,
                )
                sample.to_csv(
                    filename,
                    index=False,
                )


# Set of Asian restaurants
asian = set(alternatives[alternatives['Asian'] == 1]['ID'])

# Set of restaurants located in downtown
downtown = set(alternatives[alternatives['downtown'] == 1]['ID'])

# Set with all restaurants
all_alternatives = set(list(alternatives['ID']))

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


partition_pure = (StratumTuple(subset=all_alternatives, sample_size=SAMPLE_SIZE),)

generate_samples(name='pure', partition=partition_pure)

partition_asian = (
    StratumTuple(subset=asian, sample_size=half),
    StratumTuple(subset=all_alternatives - asian, sample_size=SAMPLE_SIZE - half),
)

generate_samples(name='asian', partition=partition_asian)

partition_downtown = (
    StratumTuple(subset=downtown, sample_size=half),
    StratumTuple(subset=all_alternatives - downtown, sample_size=SAMPLE_SIZE - half),
)

generate_samples(name='downtown', partition=partition_downtown)

partition_asian_downtown = (
    StratumTuple(subset=asian_and_downtown, sample_size=quarter),
    StratumTuple(subset=only_asian, sample_size=quarter),
    StratumTuple(subset=only_downtown, sample_size=quarter),
    StratumTuple(subset=others, sample_size=SAMPLE_SIZE - 3 * quarter),
)

generate_samples(name='both', partition=partition_asian_downtown)
