"""

biogeme.database
================

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Thu Nov 16 18:36:59 2023
"""

import pandas as pd
import numpy as np
from IPython.core.display_functions import display

from biogeme.database import Database
from biogeme.expressions import Variable, exp, bioDraws
from biogeme.expressions import TypeOfElementaryExpression
from biogeme.native_draws import description_of_native_draws, RandomNumberGeneratorTuple
from biogeme.segmentation import DiscreteSegmentationTuple
from biogeme.exceptions import BiogemeError
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
# `valuesFromDatabase`: evaluates an expression for each entry of the
# database. Takes as argument an expression, and returns a numpy
# series, long as the number of entries in the database, containing
# the calculated quantities.

# %%
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
expr = Variable1 + Variable2
result = my_data.values_from_database(expr)
print(result)

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
my_data.check_segmentation(correct_segmentation)

# %%
incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2'}
incorrect_segmentation = DiscreteSegmentationTuple(
    variable='Choice', mapping=incorrect_mapping
)

# %%
# If the segmentation is incorrect, an exception is raised.

# %%
try:
    my_data.check_segmentation(incorrect_segmentation)
except BiogemeError as e:
    print(e)

# %%
another_incorrect_mapping = {1: 'Alt. 1', 2: 'Alt. 2', 4: 'Does not exist'}
another_incorrect_segmentation = DiscreteSegmentationTuple(
    variable='Choice', mapping=another_incorrect_mapping
)

# %%
try:
    my_data.check_segmentation(another_incorrect_segmentation)
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
result = my_data.check_availability_of_chosen_alt(avail, Choice)
print(result)

# %%
# `choiceAvailabilityStatistics`: calculates the number of time an
# alternative is chosen and available.
my_data.choice_availability_statistics(avail, Choice)

# %%
# Suggest a scaling of the variables in the database
# %%
display(my_data.data.columns)

# %%
my_data.suggest_scaling()

# %%
my_data.suggest_scaling(columns=['Variable1', 'Variable2'])

# %%
# `scaleColumn`: divide an entire column by a scale value
# %%
# Before.
display(my_data.data)

# %%
my_data.scale_column('Variable2', 0.01)

# %%
# After.
display(my_data.data)

# %%
# `addColumn`: add a new column in the database, calculated from an expression.
# %%
Variable1 = Variable('Variable1')
Variable2 = Variable('Variable2')
expression = exp(0.5 * Variable2) / Variable1
# expression = Variable2 * Variable1
result = my_data.add_column(expression, 'NewVariable')
print(my_data.data['NewVariable'].tolist())

# %%
display(my_data.data)

# %%
# `split`: shuffle the data, and split the data into slices. For each
# slide, an estimation and a validation sets are generated. The
# validation set is the slice itself. The estimation set is the rest
# of the data.

# %%
dataSets = my_data.split(3)
for i in dataSets:
    print("==========")
    print("Estimation:")
    print(type(i[0]))
    print(i[0])
    print("Validation:")
    print(i[1])

# %%
# `count`: counts the number of observations that have a specific
# value in a given column.

# %%
# For instance, counts the number of entries for individual 1.
# %%
my_data.count('Person', 1)

# %%
# `remove`: removes from the database all entries such that the value
# of the expression is not 0.
# %%
exclude = Variable('Exclude')
my_data.remove(exclude)
display(my_data.data)

# %%
# `dumpOnFile`: dumps the database in a CSV formatted file.
my_data.dump_on_file()

# %%
# %%bash
# cat test_dumped.dat

# %%

# `generateDraws`: generate draws for each variable. Takes as argument
#                  a dict indexed by the names of the variables,
#                  describing the types of draws. Each of them can be
#                  a native type or any type defined by the function
#                  database.setRandomNumberGenerators, as well as the
#                  list of names of the variables that require draws
#                  to be generated.  It returns a 3-dimensional table
#                  with draws. The 3 dimensions are
#
#               1. number of individuals
#               2. number of draws
#               3. number of variables

# %%
# List of native types and their description
description_of_native_draws()

# %%
random_draws1 = bioDraws('random_draws1', 'NORMAL_MLHS_ANTI')
random_draws2 = bioDraws('random_draws2', 'UNIFORM_MLHS_ANTI')
random_draws3 = bioDraws('random_draws3', 'UNIFORMSYM_MLHS_ANTI')

# %%
# We build an expression that involves the three random variables
x = random_draws1 + random_draws2 + random_draws3
dict_of_draws = x.dict_of_elementary_expression(TypeOfElementaryExpression.DRAWS)
types = {name: expression.drawType for name, expression in dict_of_draws.items()}
print(types)

# %%
# Generation of the draws.
the_draws_table = my_data.generate_draws(
    types, ['random_draws1', 'random_draws2', 'random_draws3'], 10
)
display(the_draws_table)

# %%
# `setRandomNumberGenerators`: defines user-defined random numbers
# generators. It takes as argument a dictionary of generators. The
# keys of the dictionary characterize the name of the generators, and
# must be different from the pre-defined generators in Biogeme:
# NORMAL, UNIFORM and UNIFORMSYM. The elements of the dictionary are
# functions that take two arguments: the number of series to generate
# (typically, the size of the database), and the number of draws per
# series.

# %%
# We first define functions returning draws, given the number of
# observations, and the number of draws


# %%
# A lognormal distribution.
def log_normal_draws(sample_size: int, number_of_draws: int) -> np.ndarray:
    return np.exp(np.random.randn(sample_size, number_of_draws))


# %%
# An exponential distribution.
def exponential_draws(sample_size: int, number_of_draws: int) -> np.ndarray:
    return -1.0 * np.log(np.random.rand(sample_size, number_of_draws))


# %%
# We associate these functions with a name in a dictionary.
# %%
rnd_dict = {
    'LOGNORMAL': RandomNumberGeneratorTuple(
        generator=log_normal_draws, description='Draws from lognormal distribution'
    ),
    'EXP': RandomNumberGeneratorTuple(
        generator=exponential_draws, description='Draws from exponential distributions'
    ),
}
my_data.set_random_number_generators(rnd_dict)

# %%
# We can now generate draws from these distributions.
random_draws1 = bioDraws('random_draws1', 'LOGNORMAL')
random_draws2 = bioDraws('random_draws2', 'EXP')
x = random_draws1 + random_draws2
the_draws = x.dict_of_elementary_expression(TypeOfElementaryExpression.DRAWS)
the_types = {name: expression.drawType for name, expression in the_draws.items()}
the_draws_table = my_data.generate_draws(
    draw_types=the_types, names=['random_draws1', 'random_draws2'], number_of_draws=10
)
print(the_draws_table)

# %%
# `sampleWithReplacement`: extracts a random sample from the database,
# with replacement. Useful for bootstrapping.

# %%
my_data.sample_with_replacement()

# %%
my_data.sample_with_replacement(6)

# %%
# `panel`: defines the data as panel data. Takes as argument the name
# of the column that identifies individuals.
my_panel_data = Database('test', df)

# %%
# Data is not considered panel yet
my_panel_data.is_panel()

# %%
my_panel_data.panel('Person')

# %%
# Now it is panel.
print(my_panel_data.is_panel())

# %%
print(my_panel_data)

# %%
# When draws are generated for panel data, a set of draws is generated
# per person, not per observation.
random_draws1 = bioDraws('random_draws1', 'NORMAL')
random_draws2 = bioDraws('random_draws2', 'UNIFORM_HALTON3')

# %%
# We build an expression that involves the two random variables
x = random_draws1 + random_draws2
the_draws = x.dict_of_elementary_expression(TypeOfElementaryExpression.DRAWS)
types = {name: expression.drawType for name, expression in the_draws.items()}
the_draws_table = my_panel_data.generate_draws(
    types, ['random_draws1', 'random_draws2'], 10
)
print(the_draws_table)

# %%
# `getNumberOfObservations`: reports the number of observations in the
# database. Note that it returns the same value, irrespectively if the
# database contains panel data or not.
my_data.get_number_of_observations()

# %%
my_panel_data.get_number_of_observations()

# %%
# `getSampleSize`: reports the size of the sample. If the data is
# cross-sectional, it is the number of observations in the
# database. If the data is panel, it is the number of individuals.
my_data.get_sample_size()

# %%
my_panel_data.get_sample_size()

# %%
# `sampleIndividualMapWithReplacement`: extracts a random sample of
# the individual map from a panel data database, with
# replacement. Useful for bootstrapping.
my_panel_data.sample_individual_map_with_replacement(10)
