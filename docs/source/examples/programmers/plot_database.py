"""

biogeme.database
================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 02:30:19
"""

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from biogeme.database import (
    Database,
    PanelDatabase,
    check_availability_of_chosen_alt,
    choice_availability_statistics,
)
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Variable, exp
from biogeme.segmentation import DiscreteSegmentationTuple, verify_segmentation
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())

# %%
# We set the seed so that the outcome of random operations is always the same.
np.random.seed(90267)

# %%
# Create a database from a pandas data frame.
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 1, 1, 1],
        'Av3': [0, 1, 1, 1, 1],
    }
)
my_data = Database('test', df)
print(my_data)


# %%
# `check_segmentation`: checks that the segmentation covers the complete database.
# A segmentation is a partition of the dataset based on the value of
# one of the variables. For instance, we can segment on the Choice
# variable.

# %%
correct_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 3: 'Alt. 3'}
correct_segmentation = DiscreteSegmentationTuple(
    variable='Choice', mapping=correct_mapping
)

# %%
# If the segmentation is well-defined, the function returns the size
# of each segment in the database.

# %%
verify_segmentation(dataframe=my_data.dataframe, segmentation=correct_segmentation)

# %%
incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2'}
incorrect_segmentation = DiscreteSegmentationTuple(
    variable='Choice', mapping=incorrect_mapping
)

# %%
# If the segmentation is incorrect, an exception is raised.

# %%
try:
    verify_segmentation(
        dataframe=my_data.dataframe, segmentation=incorrect_segmentation
    )
except BiogemeError as e:
    print(e)

# %%
another_incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 4: 'Does not exist'}
another_incorrect_segmentation = DiscreteSegmentationTuple(
    variable='Choice', mapping=another_incorrect_mapping
)

# %%
try:
    verify_segmentation(
        dataframe=my_data.dataframe, segmentation=another_incorrect_segmentation
    )
except BiogemeError as e:
    print(e)

# %%
# `checkAvailabilityOfChosenAlt`: check if the chosen alternative
# is available for each entry in the database.
# %%
Av1 = Variable('Av1')
Av2 = Variable('Av2')
Av3 = Variable('Av3')
Choice = Variable('Choice')
avail = {1: Av1, 2: Av2, 3: Av3}
result = check_availability_of_chosen_alt(database=my_data, avail=avail, choice=Choice)
print(result)

# %%
# `choiceAvailabilityStatistics`: calculates the number of time an
# alternative is chosen and available.
statistics = choice_availability_statistics(
    database=my_data, avail=avail, choice=Choice
)
for alternative, choice_available in statistics.items():
    print(
        f'Alternative {alternative} is chosen {choice_available.chosen} times '
        f'and available {choice_available.available} times'
    )

# %%
# Suggest a scaling of the variables in the database
# %%
display(my_data.dataframe.columns)

# %%
suggested_scaling = my_data.suggest_scaling()
display(suggested_scaling)

# %%
# It is possible to obtain the scaling for selected variables
suggested_scaling = my_data.suggest_scaling(columns=['Variable1', 'Variable2'])
display(suggested_scaling)

# %%
# `scale_column`: divide an entire column by a scale value
# %%
# Before.
display(my_data.dataframe)

# %%
my_data.scale_column('Variable2', 0.01)

# %%
# After.
display(my_data.dataframe)

# %%
# `define_variable`: add a new column in the database, calculated from an expression.
# %%
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
expression = exp(0.5 * Variable2) / Variable1
result = my_data.define_variable(name='NewVariable', expression=expression)
print(my_data.dataframe['NewVariable'].tolist())

# %%
display(my_data.dataframe)


# %%
# `remove`: removes from the database all entries such that the value
# of the expression is not 0.
# %%
exclude = Variable('Exclude')
my_data.remove(exclude)
display(my_data.dataframe)


# %%
# `sample_with_replacement`: extracts a random sample from the database,
# with replacement. Useful for bootstrapping.

# %%
# One bootstrap sample
bootstrap_sample = my_data.bootstrap_sample()
display(bootstrap_sample.dataframe)

# %%
# Another bootstrap sample
bootstrap_sample = my_data.bootstrap_sample()
display(bootstrap_sample.dataframe)

# %%
# If the database is organised for panel data, where several observations are available for each individual, the database
# must be flattened so that each row corresponds to an individual
my_panel_data = PanelDatabase(database=my_data, panel_column='Person')
flattened_dataframe, largest_group = my_panel_data.flatten_database(
    missing_data='999999'
)
print(f'The size of the largest group of data per individual is {largest_group}')

# %%
# The name of the columns of the flat dataframe are the name of the original columns, with a suffix.
# For each variable column in the original DataFrame (excluding the column identifying the individuals),
# the output contains multiple columns named `columnname__panel__XX`, where `XX` is the zero-padded observation index
# (starting at 01). Additionally, for each observation index, a `relevant_XX` column indicates whether the observation
# is relevant (1) or padded with a missing value (0).
print('The columns of the flat dataframe are:')
for col in flattened_dataframe.columns:
    print(f'\t{col}')
display(flattened_dataframe)
