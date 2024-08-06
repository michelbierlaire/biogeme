"""

biogeme.models
==============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

:author: Michel Bierlaire
:date: Wed Nov 22 15:24:34 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biogeme.version import getText
import biogeme.database as db
from biogeme import models
from biogeme.nests import (
    OneNestForNestedLogit,
    OneNestForCrossNestedLogit,
    NestsForNestedLogit,
    NestsForCrossNestedLogit,
)
from biogeme.expressions import Variable, Beta

# %%
# Version of Biogeme.
print(getText())


# %%
# Definition of a database
# ++++++++++++++++++++++++
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [1, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 0, 1, 1],
        'Av3': [0, 1, 1, 1, 0],
    }
)
df

# %%
my_data = db.Database('test', df)

# %%
# Piecewise linear specification
# ++++++++++++++++++++++++++++++

# %%
# A piecewise linear specification (sometimes called 'spline') is a
# continuous but not differentiable function of the variable. It is
# defined based on thresholds. Between two thresholds, the function is
# linear. And the slope is changing after each threshold.
#
# Consider a variable :math:`t` and an interval :math:`[a, a+b]`. We
# define a new variable
#
# .. math:: x_{[a,b]}(t) = \max(0,\min(t-a,b)) =
#     \left\{
#        \begin{array}{ll}
#            0 & \text{if } t < a, \\
#            t-a & \text{if } a \leq t < a+b, \\
#            b  & \text{otherwise}.
#        \end{array}
#     \right.
#
# For each interval :math:`]-\infty,a]`, we have
#
# .. math:: x_{]-\infty,a]}(t) = \min(t,a) = \left\{
#     \begin{array}{ll}
#         t & \text{if } t < a, \\
#         a  & \text{otherwise}.
#      \end{array}
#      \right..
#
# For each interval :math:`[a,+\infty[`, we have
#
# .. math:: x_{]-\infty,a]}(t) = \max(0,t-a) = \left\{
#     \begin{array}{ll} 0& \text{if } t < a, \\ t-a &
#     \text{otherwise}.  \end{array} \right..
#
# If we consider a series of threshold
#
# .. math:: \alpha_1 < \alpha_2 < \ldots <\alpha_K,
#
# the piecewise linear transform of variable :math:`t` is
#
# .. math:: \sum_{k=1}^{K-1} \beta_k x_{[\alpha_k,\alpha_{k+1}]},
#
# where :math:`\beta_k` is the slope of the linear function in interval
# :math:`[\alpha_k,\alpha_{k+1}]`.
#

# %%
# The next statement generates the variables, given the thresholds. A
# `None` is equivalent to :math:`\infty`, and can only appear first (and it
# means :math:`-\infty`) or last (and it means :math:`+\infty`).
x = Variable('x')
thresholds = [None, 90, 180, 270, None]
variables = models.piecewiseVariables(x, thresholds)
print(variables)

# %%
# The next statement automatically generates the formula, including
# the Beta parameters, that are initialized to zero.
formula = models.piecewiseFormula('x', thresholds)
print(formula)

# %%
# It is also possible to initialize the Beta parameters with other
# values. Note also that the first argument can be either the name of
# the variable (as in the previous call) or the variable itself.
betas = [-0.016806308, -0.010491137, -0.002012234, -0.020051303]
formula = models.piecewiseFormula(x, thresholds, betas)
print(formula)

# %%
# We provide a plot of a piecewise linear specification.

# %%
X = np.arange(0, 300, 0.1)
Y = [
    models.piecewiseFunction(
        x, thresholds, [-0.016806308, -0.010491137, -0.002012234, -0.020051303]
    )
    for x in X
]
plt.plot(X, Y)

# %%
# Logit
# +++++

# %%
V = {1: Variable('Variable1'), 2: 0.1, 3: -0.1}
av = {1: Variable('Av1'), 2: Variable('Av2'), 3: Variable('Av3')}

# %%
# Calculation of the (log of the) logit for the three alternatives,
# based on their availability.

# %%
# Alternative 1
p1 = models.logit(V, av, 1)
p1.getValue_c(my_data, prepareIds=True)

# %%
p1 = models.loglogit(V, av, 1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# Alternative 2
p2 = models.logit(V, av, 2)
p2.getValue_c(my_data, prepareIds=True)

# %%
p2 = models.loglogit(V, av, 2)
p2.getValue_c(my_data, prepareIds=True)

# %%
# Alternative 3
p3 = models.logit(V, av, 3)
p3.getValue_c(my_data, prepareIds=True)

# %%
p3 = models.loglogit(V, av, 3)
p3.getValue_c(my_data, prepareIds=True)

# %%
# Calculation of the log of the logit for the three alternatives,
# **assuming that they are all available**.

# %%
# Alternative 1
pa1 = models.logit(V, av=None, i=1)
pa1.getValue_c(my_data, prepareIds=True)

# %%
pa1 = models.loglogit(V, av=None, i=1)
pa1.getValue_c(my_data, prepareIds=True)

# %%
# Alternative 2
pa2 = models.logit(V, av=None, i=2)
pa2.getValue_c(my_data, prepareIds=True)

# %%
pa2 = models.loglogit(V, av=None, i=2)
pa2.getValue_c(my_data, prepareIds=True)

# %%
# Alternative 3
pa3 = models.logit(V, av=None, i=3)
pa3.getValue_c(my_data, prepareIds=True)

# %%
pa3 = models.loglogit(V, av=None, i=3)
pa3.getValue_c(my_data, prepareIds=True)

# %%
# Boxcox transform
# ++++++++++++++++

# %%
# The Box-Cox transform of a variable :math:`x` is defined as
#
# .. math:: B(x,\ell) =\frac{x^{\ell}-1}{\ell},
#
# where :math:`\ell` is a parameter that can be estimated from data.  It has
# the property that
#
# .. math:: \lim_{\ell \to 0} B(x,\ell)=\log(x).
x = Variable('Variable1')
models.boxcox(x, 4)

# %%
x = Variable('Variable1')
models.boxcox(x, 0)

# %%
ell = Variable('Variable2')
e = models.boxcox(x, ell)
print(e)

# %%
e.getValue_c(my_data, prepareIds=True)

# %%
# We numerically illustrate that, when :math:`\lambda` goes to 0, the BoxCox
# transform of :math:`x` converges to the log of :math:`x`.

# %%
for ell in range(1, 16):
    x = 3
    bc = models.boxcox(x, 10**-ell).getValue()
    print(f'ell=l0^(-{ell}): {bc:.6g} - {np.log(x):.6g} = {bc - np.log(x):.6g}')


# %%
# MEV models
# ++++++++++

# %%
# MEV models are defined as
#
# .. math:: \frac{e^{V_i + \ln G_i(e^{V_1},\ldots,e^{V_J})}}{\sum_j
#     e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}},
#
# where :math:`G` is a generating function, and
#
# .. math:: G_i=\frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}).

# %%
# **Nested logit model**:
# the :math:`G` function for the nested logit model is defined such that
#
# .. math:: G_i=\frac{\partial G}{\partial
#     y_i}(e^{V_1},\ldots,e^{V_J}) = \mu e^{(\mu_m-1)V_i}
#     \left(\sum_{i=1}^{J_m} e^{\mu_m
#     V_i}\right)^{\frac{\mu}{\mu_m}-1},
#
# where the choice set is partitioned into :math:`J_m` nests, each
# associated with a parameter :math:`\mu_m`, and :math:`\mu` is the scale
# parameter. The condition is :math:`0 \leq \mu \leq \mu_m` must be
# verified. In general, :math:`\mu` is normalized to 1.0.

# %%
# This is an example with 5 alternatives. Nest A contains alternatives
# 1, 2 and 4, and is associated with a scale parameter
# :math:`\mu_A=1.2`. Nest B contains alternatives 3 and 5, and is associated
# with a scale parameter :math:`\mu_B=2.3`.

# %%
V = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}

# %%
# Definition of the nests.
nest_a = OneNestForNestedLogit(
    nest_param=1.2, list_of_alternatives=[1, 2, 4], name='nest_a'
)
nest_b = OneNestForNestedLogit(
    nest_param=2.3, list_of_alternatives=[3, 5], name='name_b'
)

nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))

# %%
p1 = models.nested(V, availability=av, nests=nests, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# If all the alternatives are available, define the availability dictionary as None.
p1 = models.nested(V, availability=None, nests=nests, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# The syntax is similar to obtain the log of the probability.
p2 = models.lognested(V, availability=av, nests=nests, choice=1)
p2.getValue_c(my_data, prepareIds=True)

# %%
p2 = models.lognested(V, availability=None, nests=nests, choice=1)
p2.getValue_c(my_data, prepareIds=True)

# %%
# If the value of the parameter :math:`\mu` is not 1, there is another
# function to call. Note that, for the sake of computational
# efficiency, it is not verified by the code if the condition :math:`0 \leq
# \mu \leq \mu_m` is valid.
p1 = models.nestedMevMu(V, availability=av, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
p1 = models.nestedMevMu(V, availability=None, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
p1 = models.lognestedMevMu(V, availability=av, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
p1 = models.lognestedMevMu(V, availability=None, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# The validity of the nested structure can be verified.
nest_c = OneNestForNestedLogit(nest_param=2.3, list_of_alternatives=[3], name='name_c')
nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))

is_valid, msg = nests.check_partition()

# %%
is_valid

# %%
print(msg)

# %%
# If an alternative belongs to two nests

# %%
nest_a = OneNestForNestedLogit(nest_param=1.2, list_of_alternatives=[1, 2, 3, 4])
nest_b = OneNestForNestedLogit(nest_param=2.3, list_of_alternatives=[3, 5])
nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))
is_valid, msg = nests.check_partition()

# %%
is_valid

# %%
print(msg)

# %%
# **Cross-nested logit model**: the :math:`G` function for the cross
# nested logit model is defined such that
#
# .. math:: G_i=\frac{\partial G}{\partial
#     y_i}(e^{V_1},\ldots,e^{V_J}) = \mu \sum_{m=1}^{M}
#     \alpha_{im}^{\frac{\mu_m}{\mu}} e^{(\mu_m-1) V_i}\left(
#     \sum_{j=1}^{J} \alpha_{jm}^{\frac{\mu_m}{\mu}} e^{\mu_m V_j}
#     \right)^{\frac{\mu}{\mu_m}-1},
#
# where each nest :math:`m`  is associated with a parameter :math:`\mu_m` and, for
# each alternative :math:`i`, a parameter :math:`\alpha_{im} \geq 0` that captures
# the degree of membership of alternative :math:`i` to nest :math:`m`. :math:`\mu` is
# the scale parameter. For each alternative :math:`i`, there must be at
# least one nest :math:`m` such that :math:`\alpha_{im}>0`. The condition :math:`0
# \leq \mu \leq \mu_m` must be also verified. In general, :math:`\mu` is
# normalized to 1.0.

# %%
# This is an example with 5 alternatives and two nests.
#
# - Alt. 1 belongs to nest A.
# - Alt. 2 belongs to nest A.
# - Alt. 3 belongs to both nest A (50%) and nest B (50%).
# - Alt. 4 belongs to nest B.
# - Alt. 5 belongs to nest B.

# %%
V = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
alpha_a = {1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
alpha_b = {1: 0, 2: 0, 3: 0.5, 4: 1, 5: 1}
nest_a = OneNestForCrossNestedLogit(
    nest_param=1.2, dict_of_alpha=alpha_a, name='Nest a'
)
nest_b = OneNestForCrossNestedLogit(
    nest_param=2.3, dict_of_alpha=alpha_b, name='Nest b'
)
nests = NestsForCrossNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))

# %%
p1 = models.cnl(V, availability=av, nests=nests, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# If all the alternatives are available, define the availability dictionary as None.
p1 = models.cnl(V, availability=None, nests=nests, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# If the value of the parameter :math:`\mu` is not 1, there is another
# function to call. Note that, for the sake of computational
# efficiency, it is not verified by the code if the condition :math:`0 \leq
# \mu \leq \mu_m` is verified.
p1 = models.cnlmu(V, availability=av, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
p1 = models.cnlmu(V, availability=None, nests=nests, choice=1, mu=1.1)
p1.getValue_c(my_data, prepareIds=True)

# %%
# If the sample is endogenous, a correction must be included in the
# model, as proposed by `Bierlaire, Bolduc and McFadden (2008)
# <http://dx.doi.org/10.1016/j.trb.2007.09.003>`_.
# In this case, the generating function must first be defined, and the
# MEV model with correction is then called.
logGi = models.getMevForCrossNested(V, availability=av, nests=nests)
logGi

# %%
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = models.mev_endogenousSampling(V, logGi, av, correction, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = models.logmev_endogenousSampling(V, logGi, av, correction, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = models.mev_endogenousSampling(V, logGi, av=None, correction=correction, choice=1)
p1.getValue_c(my_data, prepareIds=True)

# %%
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = models.logmev_endogenousSampling(
    V, logGi, av=None, correction=correction, choice=1
)
p1.getValue_c(my_data, prepareIds=True)

# %%
# The MEV generating function for the following models are available.

# %%
# Nested logit model
V = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
nest_a = OneNestForNestedLogit(
    nest_param=Beta('muA', 1.2, 1.0, None, 0), list_of_alternatives=[1, 2, 4]
)
nest_b = OneNestForNestedLogit(
    nest_param=Beta('muB', 2.3, 1.0, None, 0), list_of_alternatives=[3, 5]
)
nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))

# %%
logGi = models.getMevForNested(V, availability=None, nests=nests)
logGi

# %%
# And with the :math:`\mu` parameter.

# %%
logGi = models.getMevForNestedMu(V, availability=None, nests=nests, mu=1.1)
logGi

# %%
# Cross nested logit model

# %%
V = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
alpha_a = {1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
alpha_b = {1: 0, 2: 0, 3: 0.5, 4: 1, 5: 1}
nest_a = OneNestForCrossNestedLogit(
    nest_param=Beta('muA', 1.2, 1.0, None, 0), dict_of_alpha=alpha_a
)
nest_b = OneNestForCrossNestedLogit(
    nest_param=Beta('muB', 2.3, 1.0, None, 0), dict_of_alpha=alpha_b
)
nests = NestsForCrossNestedLogit(choice_set=list(V), tuple_of_nests=(nest_a, nest_b))
# %%
logGi = models.getMevForCrossNested(V, availability=None, nests=nests)
logGi

# %%
# Cross nested logit model with :math:`\mu` parameter.

# %%
logGi = models.getMevForCrossNestedMu(V, availability=None, nests=nests, mu=1.1)
logGi
