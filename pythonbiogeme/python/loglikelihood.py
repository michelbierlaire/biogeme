## \file
# Functions to calculate the log likelihood

from biogeme import *

## Simply computes the log of the probability
# \ingroup likelihood
# \param prob An <a
# href="http://biogeme.epfl.ch/expressions.html">expression</a>
# providing the value of the probability.
# \return The logarithm of the probability.
#
def loglikelihood(prob):
    loglikelihood = log(prob)
    return loglikelihood

## Compute a simulated loglikelihood function
# \ingroup likelihood
# \param prob An <a
# href="http://biogeme.epfl.ch/expressions.html">expression</a>
# providing the value of the probability. Although it is not formally
# necessary, the expression should contain one or more <a
# href="http://biogeme.epfl.ch/expressions.php#draws"
# target="_blank">random variables</a> of a given distribution, and
# therefore write \f[ P(i|\xi_1,\ldots,\xi_L)\f] 
# \return The simulated
# loglikelihood, given by 
# \f[ \ln\left(\sum_{r=1}^R P(i|\xi^r_1,\ldots,\xi^r_L) \right)\f]
# where \f$R\f$ is the number of draws, and \f$\xi_j^r\f$ is the
# <em>r</em>th draw of the random variable \f$\xi_j\f$.
#
def mixedloglikelihood(prob):
    l = MonteCarlo(prob)
    return log(l)

## Computes likelihood function of a regression model.
# \ingroup likelihood
# \param meas An <a href="http://biogeme.epfl.ch/expressions.html">expression</a> providing the value \f$y\f$ of the measure for the current observation.
# \param model An <a href="http://biogeme.epfl.ch/expressions.html">expression</a> providing the output \f$m\f$ of the model for the current observation.
# \param sigma An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
# (typically, a <a href="http://biogeme.epfl.ch/expressions.php#parameter">parameter</a>)
# providing the standard error \f$\sigma\f$ of the error term.
# \return The likelihood of the regression, assuming a normal
# distribution, that is
# \f[
#  \frac{1}{\sigma} \phi\left( \frac{y-m}{\sigma} \right) 
# \f]
# where \f$ \phi(\cdot)\f$ is the pdf of the normal distribution.
#
def likelihoodregression(meas,model,sigma):
    t = (meas - model) / sigma
    f = bioNormalPdf(t) / sigma
    return f


## Computes log likelihood function of a regression model.
# \ingroup likelihood
# \param meas An <a href="http://biogeme.epfl.ch/expressions.html">expression</a> providing the value \f$y\f$ of the measure for the current observation.
# \param model An <a href="http://biogeme.epfl.ch/expressions.html">expression</a> providing the output \f$m\f$ of the model for the current observation.
# \param sigma An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
# (typically, a <a href="http://biogeme.epfl.ch/expressions.php#parameter">parameter</a>)
# providing the standard error \f$\sigma\f$ of the error term.
# \return The likelihood of the regression, assuming a normal
# distribution, that is
# \f[
#  -\left( \frac{(y-m)^2}{2\sigma^2} \right) - \log(\sigma) - \frac{1}{2}\log(2\pi)
# \f]
# 
#
def loglikelihoodregression(meas,model,sigma):
    t = (meas - model) / sigma
    f = - t * t / 2 - log(sigma) -0.9189385332
    return f


