## \file 
# Implementation of the pdf and CDF of common distributions
# \author Michel Bierlaire
# \date Thu Apr 23 12:01:49 2015
from biogeme import *

## \brief Normal pdf
#
# Probability density function of a normal distribution \f[ f(x;\mu,\sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp{-\frac{(x-\mu)^2}{2\sigma^2}} \f]
# \ingroup pdfcdf
# \param x argument of the pdf 
# \param mu location parameter \f$\mu\f$ (Default: 0)
# \param s scale parameter \f$\sigma\f$ (Default: 1)
# \note It is assumed that \f$\sigma > 0\f$, but it is not verified by the code.
def normalpdf(x,mu=0.0,s=1.0):
    d = -(x-mu)*(x-mu)
    n = 2.0*s*s
    a = d/n
    num = exp(a)
    den = s*2.506628275
    p = num / den
    return p

## \brief Log normal pdf
#
# Probability density function of a log normal distribution \f[ f(x;\mu,\sigma) =  \frac{1}{x\sigma \sqrt{2\pi}} \exp{-\frac{(\ln x-\mu)^2}{2\sigma^2}} \f]
# \ingroup pdfcdf
# \param x argument of the pdf 
# \param mu location parameter \f$\mu\f$ (Default: 0)
# \param s scale parameter \f$\sigma\f$ (Default: 1)
# \note It is assumed that \f$\sigma > 0\f$, but it is not verified by the code.
def lognormalpdf(x,mu,s):
    d = -(log(x)-mu)*(log(x)-mu)
    n = 2.0*s*s
    a = d/n
    num = exp(a)
    den = x*s*2.506628275
    p = (x>0)* num / den
    return p

## \brief Uniform pdf
#
# Probability density function of a uniform distribution \f[ f(x;a,b) = \left\{ \begin{array}{ll} \frac{1}{b-a} & \text{for } x \in [a,b] \\ 0 & \text{otherwise}\end{array} \right.\f]
# \ingroup pdfcdf
# \param x argument of the pdf 
# \param a lower bound \f$a\f$ of the distribution (Default: -1)
# \param b upper bound \f$b\f$ of the distribution (Default: 1)
# \note It is assumed that \f$a < b \f$, but it is not verified by the code.
def uniformpdf(x,a=-1,b=1.0):
    result = (x < a) * 0.0 + (x >= b) * 0.0 + (x >= a) * (x < b) / (b-a)
    return result

## \brief Triangular pdf
#
# Probability density function of a triangular distribution \f[ f(x;a,b,c) = \left\{ \begin{array}{ll} 0 & \text{if } x < a \\\frac{2(x-a)}{(b-a)(c-a)} & \text{if } a \leq x < c \\\frac{2(b-x)}{(b-a)(b-c)} & \text{if } c \leq x < b \\0 & \text{if } x \geq b.\end{array} \right.\f]
# \ingroup pdfcdf
# \param x argument of the pdf 
# \param a lower bound \f$a\f$ of the distribution (Default: -1)
# \param b upper bound \f$b\f$ of the distribution (Default: 1)
# \param c mode \f$c\f$ of the distribution (Default: 0)
# \note It is assumed that \f$a < b \f$, and \f$a \leq c \leq b\f$, but it is not verified by the code.
def triangularpdf(x,a=-1.0,b=1.0,c=0.0):
    result = (x < a) * 0.0 + (x >= b) * 0.0 + (x >= a) * (x < c) * 2.0 * ((x-a)/((b-a)*(c-a))) *  (x >= c) * (x < b)  * 2.0 * (b-x) / ((b-a)*(b-c))
    return result

## \brief Logistic CDF
#
# Cumulative distribution function of a logistic distribution \f[ f(x;\mu,\sigma) = \frac{1}{1+\exp\left(-\frac{x-\mu}{\sigma} \right)} \f]
# \ingroup pdfcdf
# \param x argument of the pdf 
# \param mu location parameter \f$\mu\f$ (Default: 0)
# \param s scale parameter \f$\sigma\f$ (Default: 1)
# \note It is assumed that \f$\sigma > 0\f$, but it is not verified by the code.
def logisticcdf(x,mu=0.0,s=1.0):
    result = 1.0 /( 1.0 + exp(-(x-mu)/s))
    return result
