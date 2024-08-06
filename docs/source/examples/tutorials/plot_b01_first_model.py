"""

Estimation of a binary logit model
==================================

Example extracted from Ben-Akiva and Lerman (1985)

:author: Michel Bierlaire, EPFL
:date: Thu May 16 11:59:49 2024

"""

import pandas as pd
from biogeme.database import Database
from biogeme.expressions import Variable, Beta
from biogeme.models import loglogit
from biogeme.biogeme import BIOGEME


# %%
# The data set is organized as a Pandas data frame. In this simple example, the data is provided directly in the
# script. Most of the time, the data is available from a file, or an external database, and must be imported into
# Pandas.
data = {
    'ID': pd.Series([i + 1 for i in range(21)]),
    'auto_time': pd.Series(
        [
            52.9,
            4.1,
            4.1,
            56.2,
            51.8,
            0.2,
            27.6,
            89.9,
            41.5,
            95.0,
            99.1,
            18.5,
            82.0,
            8.6,
            22.5,
            51.4,
            81.0,
            51.0,
            62.2,
            95.1,
            41.6,
        ]
    ),
    'transit_time': pd.Series(
        [
            4.4,
            28.5,
            86.9,
            31.6,
            20.2,
            91.2,
            79.7,
            2.2,
            24.5,
            43.5,
            8.4,
            84.0,
            38.0,
            1.6,
            74.1,
            83.8,
            19.2,
            85.0,
            90.1,
            22.2,
            91.5,
        ]
    ),
    'choice': pd.Series(
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    ),
}
pandas_dataframe = pd.DataFrame(data)

# %%
# The data frame is used to initialize the Biogeme database.
biogeme_database = Database('ben_akiva_lerman', pandas_dataframe)

# %%
# The next step is to provide the model specification:
#
# - the explanatory variables,
# - the parameters to be estimated,
# - the specification of the utility functions,
# - the specification of the choice model.

# %%
# Explanatory variables: the object `Variable` associates the name of a column in the database with a Python variable,
# that will be used in the utility specification. In this example, we have three variables (two independent, and
# one dependent, that is, the choice).
auto_time = Variable('auto_time')
transit_time = Variable('transit_time')
choice = Variable('choice')

# %%
# Parameters to be estimated: the object `Beta` identifies the parameters to be estimated. It accepts 5 arguments:
#
# - the name of the parameter (used for reporting),
# - the initial value (used as a starting point by the optimization algorithm),
# - a lower bound on its value, or None if there is no such bound,
# - an upper bound on its value, or None if there is no such bound,
# - a status, which is either 0 or 1. If zero, it means that the value of the parameters will be estimated. If one, it
#   means that the value will stay fixed to the initial value provided.
#
# Although not formally necessary, It is good practice to use the exact same name for the Python variable and the
# parameter itself.

# %%
# First, we define the alternative specific constant. We estimate the one associated with the car alternative. The
# one associated with transit is normalized to zero and, therefore, does not appear in the model.
asc_car = Beta('asc_car', 0, None, None, 0)

# %%
# Second, we define the coefficient of travel time.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# We are now ready to specify the utility functions.
utility_car = asc_car + b_time * auto_time
utility_transit = b_time * transit_time

# %%
# Next, we need to associate the utility function with the ID of the alternative. It is necessary to interpret
# correctly the value of the variable `choice`. We use a Python dictionary to do that.
utilities = {0: utility_car, 1: utility_transit}

# %%
# To finish the specification of the model, we need to provide an expression for the contribution to the log-likelihood
# function of each observation. As this is typically the logarithm of the choice probability, we need to select
# a choice model. In this, we select the logit model. We use the `loglogit` model to obtain the logarithm of the
# choice probability. It takes three arguments:
#
# - a dictionary with the specification of the utility functions,
# - a dictionary with the availability conditions. In this simple example, both alternatives are always available,
#   so that there is no ned to provide it,
# - the choice variable.
log_choice_probability = loglogit(utilities, None, choice)

# %%
# All the ingredients are now ready. We put them together into the `BIOGEME` object. We create by proving both the
# database and the model specification.
biogeme_object = BIOGEME(biogeme_database, log_choice_probability)

# %%
# It is recommended to provide a name to the model. Indeed, the estimation results will be saved in two files: a
# "human-readable" HTML file, and a Python-specific format called `pickle` so that existing estimation results can
# be read from file instead of being re-estimated.
biogeme_object.modelName = 'first_model'

# %%
# Finally, we run the estimation algorithm to obtain the estimates of the coefficients.
results = biogeme_object.estimate()

# %%
# The `results` object contains a great deal of information. In particular, it provides a summary of the
# estimation results.
print(results.short_summary())

# %% It can also provides the estimates of the parameters, with some statistics.
results.get_estimated_parameters()
