## \file
# Function implementing the Box-Cox transform of a variable
from biogeme import *

## Implements the Box-Cox transform of a variable
# @ingroup specs
# @param x variable to be transformed
# @param lambda \f$\lambda\f$ parameter of the transformation
# @return The Box-Cox transform, that is
#  \f[ \frac{x^\lambda - 1 }{\lambda} \f] 
def boxcox(x,l):
    return (x**l - 1) / l
