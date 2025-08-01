"""

biogeme.distributions
=====================

Example of usage of the `distributions` module.  This is for programmers who need
examples of use of the functions of the class. The examples are
designed to illustrate the syntax.

Michel Bierlaire
Sun Jun 29 2025, 07:12:12
"""

from IPython.core.display_functions import display

from biogeme.distributions import (
    logisticcdf,
    lognormalpdf,
    normalpdf,
    triangularpdf,
    uniformpdf,
)
from biogeme.expressions import Beta
from biogeme.version import get_text

print(get_text())

# %%
# pdf of the normal distributio: returns the biogeme expression of the
# probability density function of the normal distribution:
#
# .. math:: f(x;\mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}}
#           \exp{-\frac{(x-\mu)^2}{2\sigma^2}}.

# %%
# Calculated for a numeric value.
resulting_expression = normalpdf(0)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# Calculated for an expression.
a_parameter = Beta('a_parameter', 0, None, None, 1)
mu = Beta('mu', 0, None, None, 1)
sigma = Beta('sigma', 1, None, None, 1)

# %%
resulting_expression = normalpdf(a_parameter, mu=mu, s=sigma)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# pdf of the lognormal distribution: returns the biogeme expression of
# the probability density function of the lognormal distribution
#
# .. math:: f(x;\mu, \sigma) = \frac{1}{x\sigma \sqrt{2\pi}}
#               \exp{-\frac{(\ln x-\mu)^2}{2\sigma^2}}

# %%
# Calculated for a numeric value.
resulting_expression = lognormalpdf(1)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# Calculated for an expression.
a_parameter = Beta('a_parameter', 1, None, None, 1)
mu = Beta('mu', 0, None, None, 1)
sigma = Beta('sigma', 1, None, None, 1)

# %%
resulting_expression = lognormalpdf(a_parameter, mu=mu, s=sigma)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# pdf of the uniform distribution: returns the biogeme expression of
# the probability density function of the uniform distribution
#
# .. math:: f(x; a, b) = \left\{ \begin{array}{ll}
#               \frac{1}{b-a} & \mbox{for } x \in [a, b] \\
#               0 & \mbox{otherwise}\end{array} \right.

# %%
# Calculated for a numeric value
resulting_expression = uniformpdf(0)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# Calculated for an expression
a_parameter = Beta('a_parameter', 0, None, None, 1)
a = Beta('a', -1, None, None, 1)
b = Beta('b', 1, None, None, 1)

# %%
resulting_expression = uniformpdf(a_parameter, a=a, b=b)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# pdf of the triangular distribution: returns the biogeme expression
# of the probability density function of the triangular distribution
#
#
# .. math:: f(x;a, b, c) = \left\{ \begin{array}{ll} 0 &
#              \text{if } x < a \\\frac{2(x-a)}{(b-a)(c-a)} &
#              \text{if } a \leq x < c \\\frac{2(b-x)}{(b-a)(b-c)} &
#              \text{if } c \leq x < b \\0 & \text{if } x \geq b.
#              \end{array} \right.
#
# It is assumed that :math:`a < c < b`. It is not verified.

# %%
# Calculated for a numeric value.
resulting_expression = triangularpdf(0)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# Calculated for an expression.
a_parameter = Beta('a_parameter', 0, None, None, 1)
a = Beta('a', -1, None, None, 1)
b = Beta('b', 1, None, None, 1)
c = Beta('c', 0, None, None, 1)

# %%
resulting_expression = triangularpdf(a_parameter, a=a, b=b, c=c)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# CDF of the logistic distribution: returns the biogeme expression of
# the cumulative distribution function of the logistic distribution
#
#
# .. math:: f(x;\mu, \sigma) = \frac{1}{1+\exp\left(-\frac{x-\mu}{\sigma} \right)}
#
#

# %%
# Calculated for a numeric value
resulting_expression = logisticcdf(0)
display(resulting_expression)

# %%
resulting_expression.get_value()

# %%
# Calculated for an expression
a_parameter = Beta('a_parameter', 0, None, None, 1)
mu = Beta('mu', 0, None, None, 1)
sigma = Beta('sigma', 1, None, None, 1)

# %%
resulting_expression = logisticcdf(a_parameter, mu=mu, s=sigma)
display(resulting_expression)

# %%
resulting_expression.get_value()
