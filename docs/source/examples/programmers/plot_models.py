"""

biogeme.models
==============

Examples of use of several functions.

This is designed for programmers who need examples of use of the
functions of the module. The examples are designed to illustrate the
syntax. They do not correspond to any meaningful model.

Michel Bierlaire
Sun Jun 29 2025, 11:21:51
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.jax_calculator import evaluate_simple_expression_per_row
from biogeme.models import (
    boxcox,
    cnl,
    cnlmu,
    get_mev_for_cross_nested,
    get_mev_for_cross_nested_mu,
    get_mev_for_nested,
    get_mev_for_nested_mu,
    logit,
    loglogit,
    logmev_endogenous_sampling,
    lognested,
    lognested_mev_mu,
    mev_endogenous_sampling,
    nested,
    nested_mev_mu,
    piecewise_formula,
    piecewise_function,
    piecewise_variables,
)
from biogeme.nests import (
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    OneNestForNestedLogit,
)
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.version import get_text

# %%
# Version of Biogeme.
print(get_text())


# %%
# Definition of a database
# ++++++++++++++++++++++++
df = pd.DataFrame(
    {
        'Person': [1, 1, 1, 2, 2],
        'Exclude': [0, 0, 1, 0, 1],
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [10, 20, 30, 40, 50],
        'Choice': [2, 2, 3, 1, 2],
        'Av1': [0, 1, 1, 1, 1],
        'Av2': [1, 1, 0, 1, 1],
        'Av3': [0, 1, 1, 1, 0],
    }
)
display(df)

# %%
my_data = Database('test', df)

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
variables = piecewise_variables(x, thresholds)
print(variables)

# %%
# The next statement automatically generates the formula, including
# the Beta parameters, that are initialized to zero.
formula = piecewise_formula('x', thresholds)
print(formula)

# %%
# It is also possible to initialize the Beta parameters with other
# values. Note also that the first argument can be either the name of
# the variable (as in the previous call) or the variable itself.
betas = [-0.016806308, -0.010491137, -0.002012234, -0.020051303]
formula = piecewise_formula(x, thresholds, betas)
print(formula)

# %%
# We provide a plot of a piecewise linear specification.

# %%
X = np.arange(0, 300, 0.1)
Y = [
    piecewise_function(
        x, thresholds, [-0.016806308, -0.010491137, -0.002012234, -0.020051303]
    )
    for x in X
]
plt.plot(X, Y)

# %%
# Logit
# +++++

# %%
v = {1: Variable('Variable1'), 2: 0.1, 3: -0.1}
av = {1: Variable('Av1'), 2: Variable('Av2'), 3: Variable('Av3')}

# %%
# Calculation of the (log of the) logit for the three alternatives,
# based on their availability.

# %%
# Alternative 1
p1 = logit(v, av, 1)
prob_1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Probability of alternative 1: {prob_1_value}')

# %%
# Alternative 2
p2 = logit(v, av, 2)
prob_2_value = evaluate_simple_expression_per_row(
    expression=p2,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Probability of alternative 2: {prob_2_value}')

# %%
# Alternative 3
p3 = logit(v, av, 3)
prob_3_value = evaluate_simple_expression_per_row(
    expression=p3,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Probability of alternative 3: {prob_3_value}')

# %%
# Calculation of the log of the logit for the three alternatives.
# If `av` is set to None, it means that all alternatives are always available.
# %%
# Alternative 1
p1 = loglogit(util=v, av=None, i=1)
log_prob_1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Log probability of alternative 1: {log_prob_1_value}')

# %%
# Alternative 2
p2 = loglogit(util=v, av=None, i=2)
log_prob_2_value = evaluate_simple_expression_per_row(
    expression=p2,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Log probability of alternative 2: {log_prob_2_value}')

# %%
# Alternative 3
p3 = loglogit(util=v, av=None, i=3)
log_prob_3_value = evaluate_simple_expression_per_row(
    expression=p3,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Log probability of alternative 3: {log_prob_3_value}')

# %%
# Box-Cox transform
# +++++++++++++++++

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
display(boxcox(x, 4))

# %%
x = Variable('Variable1')
display(boxcox(x, 0))

# %%
ell = Variable('Variable2')
e = boxcox(x, ell)
display(e)

# %%
boxcox_value = evaluate_simple_expression_per_row(
    expression=e,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Box-Cox transform of Variable2: {boxcox_value}')

# %%
# We numerically illustrate that, when :math:`\lambda` goes to 0, the BoxCox
# transform of :math:`x` converges to the log of :math:`x`.

# %%
for ell in range(1, 16):
    x = 3
    bc = boxcox(x, 10**-ell).get_value()
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
v = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}

# %%
# Definition of the nests.
nest_a = OneNestForNestedLogit(
    nest_param=1.2, list_of_alternatives=[1, 2, 4], name='nest_a'
)
nest_b = OneNestForNestedLogit(
    nest_param=2.3, list_of_alternatives=[3, 5], name='name_b'
)

nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))

# %%
p1 = nested(v, availability=av, nests=nests, choice=1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Nested logit probability of alternative 1: {p1_value}')

# %%
# If all the alternatives are available, define the availability dictionary as None.
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(
    f'Nested logit probability of alternative 1, all alternatives are available: {p1_value}'
)

# %%
# The syntax is similar to obtain the log of the probability.
p2 = lognested(v, availability=av, nests=nests, choice=1)
p2_value = evaluate_simple_expression_per_row(
    expression=p2,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Nested logit log probability of alternative 1: {p2_value}')

# %%
p2 = lognested(v, availability=None, nests=nests, choice=1)
p2_value = evaluate_simple_expression_per_row(
    expression=p2,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(
    f'Nested logit log probability of alternative 1, all alternatives are available: {p2_value}'
)

# %%
# If the value of the parameter :math:`\mu` is not 1, there is another
# function to call. Note that, for the sake of computational
# efficiency, it is not verified by the code if the condition :math:`0 \leq
# \mu \leq \mu_m` is valid.
p1 = nested_mev_mu(v, availability=av, nests=nests, choice=1, mu=1.1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Nested logit probability of alternative 1, mu=1.1: {p1_value}')

# %%
p1 = lognested_mev_mu(v, availability=av, nests=nests, choice=1, mu=1.1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Nested logit log probability of alternative 1, mu=1.1: {p1_value}')

# %%
# The validity of the nested structure can be verified.
nest_c = OneNestForNestedLogit(nest_param=2.3, list_of_alternatives=[3], name='name_c')
nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))

is_valid, msg = nests.check_partition()

# %%
display(is_valid)

# %%
print(msg)

# %%
# If an alternative belongs to two nests

# %%
nest_a = OneNestForNestedLogit(nest_param=1.2, list_of_alternatives=[1, 2, 3, 4])
nest_b = OneNestForNestedLogit(nest_param=2.3, list_of_alternatives=[3, 5])
nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))
is_valid, msg = nests.check_partition()

# %%
display(is_valid)

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
v = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
alpha_a = {1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
alpha_b = {1: 0, 2: 0, 3: 0.5, 4: 1, 5: 1}
nest_a = OneNestForCrossNestedLogit(
    nest_param=1.2, dict_of_alpha=alpha_a, name='Nest a'
)
nest_b = OneNestForCrossNestedLogit(
    nest_param=2.3, dict_of_alpha=alpha_b, name='Nest b'
)
nests = NestsForCrossNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))

# %%
p1 = cnl(v, availability=av, nests=nests, choice=1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Cross-Nested logit probability of alternative 1: {p1_value}')

# %%
# If all the alternatives are available, define the availability dictionary as None.
p1 = cnl(v, availability=None, nests=nests, choice=1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(
    f'Cross-Nested logit probability of alternative 1, all alternatives are available: {p1_value}'
)

# %%
# If the value of the parameter :math:`\mu` is not 1, there is another
# function to call. Note that, for the sake of computational
# efficiency, it is not verified by the code if the condition :math:`0 \leq
# \mu \leq \mu_m` is verified.
p1 = cnlmu(v, availability=av, nests=nests, choice=1, mu=1.1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'Cross-Nested logit probability of alternative 1, mu = 1.1: {p1_value}')

# %%
# If the sample is endogenous, a correction must be included in the
# model, as proposed by `Bierlaire, Bolduc and McFadden (2008)
# <http://dx.doi.org/10.1016/j.trb.2007.09.003>`_.
# In this case, the generating function must first be defined, and the
# MEV model with correction is then called.
log_gi = get_mev_for_cross_nested(v, availability=av, nests=nests)
display(log_gi)

# %%
# Assume the following correction factors
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = mev_endogenous_sampling(v, log_gi, av, correction, choice=1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'MEV model probability with correction: {p1_value}')

# %%
correction = {1: -0.1, 2: 0.1, 3: 0.2, 4: -0.2, 5: 0}
p1 = logmev_endogenous_sampling(v, log_gi, av, correction, choice=1)
p1_value = evaluate_simple_expression_per_row(
    expression=p1,
    database=my_data,
    numerically_safe=False,
    second_derivatives_mode=SecondDerivativesMode.NEVER,
    use_jit=True,
)
display(f'MEV model log probability with correction: {p1_value}')

# %%
# The MEV generating function for the following models are available.

# %%
# Nested logit model
v = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
nest_a = OneNestForNestedLogit(
    nest_param=Beta('muA', 1.2, 1.0, None, 0), list_of_alternatives=[1, 2, 4]
)
nest_b = OneNestForNestedLogit(
    nest_param=Beta('muB', 2.3, 1.0, None, 0), list_of_alternatives=[3, 5]
)
nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))

# %%
log_gi = get_mev_for_nested(v, availability=None, nests=nests)
display(log_gi)

# %%
# And with the :math:`\mu` parameter.

# %%
log_gi = get_mev_for_nested_mu(v, availability=None, nests=nests, mu=1.1)
display(log_gi)

# %%
# Cross nested logit model

# %%
v = {1: Variable('Variable1'), 2: 0.1, 3: -0.1, 4: -0.2, 5: 0.2}
av = {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
alpha_a = {1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
alpha_b = {1: 0, 2: 0, 3: 0.5, 4: 1, 5: 1}
nest_a = OneNestForCrossNestedLogit(
    nest_param=Beta('muA', 1.2, 1.0, None, 0), dict_of_alpha=alpha_a
)
nest_b = OneNestForCrossNestedLogit(
    nest_param=Beta('muB', 2.3, 1.0, None, 0), dict_of_alpha=alpha_b
)
nests = NestsForCrossNestedLogit(choice_set=list(v), tuple_of_nests=(nest_a, nest_b))
# %%
log_gi = get_mev_for_cross_nested(v, availability=None, nests=nests)
display(log_gi)

# %%
# Cross nested logit model with :math:`\mu` parameter.

# %%
log_gi = get_mev_for_cross_nested_mu(v, availability=None, nests=nests, mu=1.1)
display(log_gi)
