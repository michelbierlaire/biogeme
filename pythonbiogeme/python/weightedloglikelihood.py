## \file 
# Log likelihood function for WESML

from biogeme import *

## Computes the log likelihood function for the WESML estimator. 
# @ingroup likelihood
# @param prob <a href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries" target="_blank">dictionary</a> were the keys are the identifiers of the alternatives in the choice set, and the values are expressions representing the choice probabilities.
# @param choice expression producing the id of the chosen alternative.
# @param weight expression producing the id of the chosen alternative.
# @return value of the weighted log likelihood function 
def weightedloglikelihood(prob,choice,weight):
    BIOGEME_OBJECT.WEIGHT = weight
    loglikelihood = log(Elem(prob,choice))
    return loglikelihood

