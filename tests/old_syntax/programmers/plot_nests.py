"""

biogeme.nests
=============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Wed Nov 29 18:35:06 2023
"""
import numpy as np
from biogeme.version import getText
from biogeme.nests import (
    OneNestForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
)


# %%
# Version of Biogeme.
print(getText())


# %%
# Covariance and correlation between two alternatives of a
# cross-nested logit model. Here, we test a logit model by setting the
# nest parameters to 1.0. We expect the identify matrix as correlation.
choice_set = ['i', 'j', 'k', 'm']
mu_nest_1 = 1.0
alphas_1 = {'i': 1, 'j': 1}
nest_1 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_1, dict_of_alpha=alphas_1, name='Nest 1'
)
mu_nest_2 = 1.0
alphas_2 = {'j': 0.0, 'k': 1, 'm': 1}
nest_2 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_2, dict_of_alpha=alphas_2, name='Nest 2'
)
nests = NestsForCrossNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))


# %%
nests.correlation()

# %%
# Entries of the covariance matrix can also be obtained. Here, we
# report the variance for alternative `i`.
nests.covariance('i', 'i')

# %%
# It is :math:`\pi^2/6`.
np.pi**2 / 6

# %%
# Second, a nested logit model
mu_nest_1 = 1.5
alphas_1 = {'i': 1, 'j': 1}
nest_1 = OneNestForNestedLogit(
    nest_param=mu_nest_1, list_of_alternatives=['i', 'j'], name='Nest 1'
)
mu_nest_2 = 2.0
alphas_2 = {'k': 1, 'm': 1}
nest_2 = OneNestForNestedLogit(
    nest_param=mu_nest_2, list_of_alternatives=['k', 'm'], name='Nest 2'
)
nests = NestsForNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))

# %%
nests.correlation()

# %%
# Theoretical value for the correlation
correl_nest_1 = 1 - (1 / mu_nest_1**2)
correl_nest_1

# %%
correl_nest_2 = 1 - (1 / mu_nest_2**2)
correl_nest_2


# %%
# The same nested logit model, coded as a cross-nested logit


mu_nest_1 = 1.5
alphas_1 = {'i': 1, 'j': 1}
nest_1 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_1, dict_of_alpha=alphas_1, name='Nest 1'
)
mu_nest_2 = 2.0
alphas_2 = {'j': 0.0, 'k': 1, 'm': 1}
nest_2 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_2, dict_of_alpha=alphas_2, name='Nest 2'
)
nests = NestsForCrossNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))


# %%
nests.correlation()

# %%
# Finally, a cross-nested logit model, where alternative j is
# correlated with all the other alternatives, and belong to two
# different nests.
mu_nest_1 = 1.5
alphas_1 = {'i': 1, 'j': 0.5}
nest_1 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_1, dict_of_alpha=alphas_1, name='Nest 1'
)
mu_nest_2 = 2.0
alphas_2 = {'j': 0.5, 'k': 1, 'm': 1}
nest_2 = OneNestForCrossNestedLogit(
    nest_param=mu_nest_2, dict_of_alpha=alphas_2, name='Nest 2'
)
nests = NestsForCrossNestedLogit(choice_set=choice_set, tuple_of_nests=(nest_1, nest_2))

# %%
nests.correlation()
