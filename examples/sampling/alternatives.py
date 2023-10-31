"""Script reading the list of alternatives and identifying subsets

Michel Bierlaire
Mon Oct  9 10:53:03 2023
"""
import pandas as pd

alternatives = pd.read_csv('restaurants.dat')
ID_COLUMN = 'ID'

all_alternatives = set(list(alternatives[ID_COLUMN]))

# Set of Asian restaurants
asian = set(alternatives[alternatives['Asian'] == 1][ID_COLUMN])
print(f'Number of asian restaurants: {len(asian)}')

# Set of restaurants located in downtown
downtown = set(alternatives[alternatives['downtown'] == 1][ID_COLUMN])

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


def complement(a_set: set[int]) -> set[int]:
    """Returns the complement of a set"""
    return all_alternatives - a_set
