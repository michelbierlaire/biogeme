"""

Configuring Biogeme with parameters
===================================

We illustrate how to obtain information about configuration parameters, and how to modify them.

Michel Bierlaire, EPFL
Thu May 16 13:24:56 2024

"""

import os

import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.default_parameters import print_list_of_parameters
from biogeme.expressions import Beta

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Illustration of the definition of parameters')

# %%
# Biogeme accepts several parameters that modifies its functionalities. In this example, we illustrate
# how to obtain information about those parameters, and how to modify them.

# %%
# We first create a dummy dataset that is needed to create the Biogeme object. Its content is irrelevant.
data = {
    'x1': pd.Series([0]),
}
pandas_dataframe = pd.DataFrame(data)
biogeme_database = Database('dummy', pandas_dataframe)

# %%
# We also create a dummy model, irrelevant as well.
logprob = Beta('dummy_parameter', 0, None, None, 0)

# %%
# When you create the Biogeme model, Biogeme tries to read the values of the parameters
# from the file `biogeme.toml`. If the file does not exist, default values are used.
biogeme_object = BIOGEME(database=biogeme_database, formulas=logprob)

# %%
# For instance, let's check the value for the maximum number of iterations:
print(f'Max. number of iterations: {biogeme_object.max_iterations}')

# %%
# If it did not exist before, Biogeme has created a file called `biogeme.toml`.
default_toml_file_name = 'biogeme.toml'
with open(default_toml_file_name, 'r') as file:
    lines = file.readlines()

# %%
# Here are the first lines of this file. As you see, it is structured into sections. Each section contains
# a list of parameters, their value, and a short description.
number_of_lines_to_display = 20
for i, line in enumerate(lines):
    print(line, end='')
    if i == number_of_lines_to_display:
        break

# %%
# Let's now replace the value of a parameter in the file by 500.
for i, line in enumerate(lines):
    if 'max_iterations' in line:
        lines[i] = 'max_iterations = 500\n'

with open(default_toml_file_name, 'w') as file:
    file.writelines(lines)

# %%
# We create a new Biogeme object. The values of the parameters are read from
# the file `biogeme.toml`.
biogeme_object = BIOGEME(database=biogeme_database, formulas=logprob)

# %%
# We check that the value 500 that we have specified has indeed been considered.
print(f'Max. number of iterations: {biogeme_object.max_iterations}')

# %%
# It is possible to have several toml files, with different configurations. For instance, let's create another
# file with a different value for the `max_iterations` parameter: 650.
another_toml_file_name = 'customized.toml'
new_value = 650
for i, line in enumerate(lines):
    if 'max_iterations' in line:
        lines[i] = f'max_iterations = {new_value}\n'
with open(another_toml_file_name, 'w') as file:
    file.writelines(lines)

# %%
# The name of the file must now be specified at the creation of the Biogeme object.
biogeme_object = BIOGEME(
    database=biogeme_database, formulas=logprob, parameters=another_toml_file_name
)
print(f'Max. number of iterations: {biogeme_object.max_iterations}')

# %%
# Note that if you specify the name of a file that does not exist, this file will be created, and the default value
# of the parameters used.
yet_another_toml_file_name = 'xxx.toml'
biogeme_object = BIOGEME(
    database=biogeme_database, formulas=logprob, parameters=yet_another_toml_file_name
)
print(f'Max. number of iterations: {biogeme_object.max_iterations}')

# %%
# Another way to set the value of a parameter is to specify it explicitly at the creation of the Biogeme object.
# It supersedes the value in the .toml file.
yet_another_value = 234
biogeme_object = BIOGEME(
    database=biogeme_database, formulas=logprob, max_iterations=yet_another_value
)
print(f'Max. number of iterations: {biogeme_object.max_iterations}')

# %%
# Both can be combined.
biogeme_object = BIOGEME(
    database=biogeme_database,
    formulas=logprob,
    max_iterations=234,
    parameters=another_toml_file_name,
)
print(f'Max. number of iterations: {biogeme_object.max_iterations}')
# %%
# We delete the toml files to clean the directory.
os.remove(default_toml_file_name)
os.remove(another_toml_file_name)
os.remove(yet_another_toml_file_name)

# %%
# Finally, we display the list of all parameters
display(print_list_of_parameters())
