"""

List of alternatives
====================

Script reading the list of alternatives and identifying subsets

:author: Michel Bierlaire
:date: Mon Oct  9 10:53:03 2023
"""

import pandas as pd
from biogeme.partition import Partition

# %%
alternatives = pd.read_csv('restaurants.dat')
alternatives

# %%
ID_COLUMN = 'ID'

# %%
all_alternatives = set(list(alternatives[ID_COLUMN]))

# %%
# Set of Asian restaurants
asian = set(alternatives[alternatives['Asian'] == 1][ID_COLUMN])
print(f'Number of asian restaurants: {len(asian)}')

# %%
# Set of restaurants located in downtown
downtown = set(alternatives[alternatives['downtown'] == 1][ID_COLUMN])

# %%
# Set of Asian restaurants in downtown
asian_and_downtown = asian & downtown

# %%
# Set of Asian restaurants, and of restaurants in downtown
asian_or_downtown = asian | downtown

# %%
# Set of Asian restaurants not in downtown
only_asian = asian - asian_and_downtown

# %%
# Set of non Asian restaurants in downtown
only_downtown = downtown - asian_and_downtown

# %%
# Set of restaurants that are neither Asian nor in downtown
others = all_alternatives - asian_or_downtown


# %%
def complement(a_set: set[int]) -> set[int]:
    """Returns the complement of a set"""
    return all_alternatives - a_set


# %%
# Partitions.
partition_asian = Partition([asian, complement(asian)], full_set=all_alternatives)
partition_downtown = Partition(
    [downtown, complement(downtown)], full_set=all_alternatives
)
partition_uniform = Partition([all_alternatives], full_set=all_alternatives)
partition_uniform_asian = Partition([asian], full_set=asian)
partition_uniform_asian_or_downtown = Partition(
    [asian_or_downtown], full_set=asian_or_downtown
)

# %%
partitions = {
    'uniform': partition_uniform,
    'asian': partition_asian,
    'downtown': partition_downtown,
    'uniform_asian_or_downtown': partition_uniform_asian_or_downtown,
    'uniform_asian': partition_uniform_asian,
}
