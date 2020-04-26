## \file 
# Functions calculating statistics on the sample

from biogeme import *

## Computes the null log likelihood from the sample and ask Biogeme to include it in the output file.
# \ingroup stats
# \param availability A <a href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries" target="_blank">dictionary</a> mapping each alternative ID with its availability condition.
# \param iterator An iterator on the data file.
# \return log likelihood of a model where the choice probability for
# observation \f$n\f$ is given by is \f$1/J_n\f$, where \f$J_n\f$ is
# the number of available alternatives, i.e. \f[ \mathcal{L} = -\sum_n \ln(J_n) \f]
def nullLoglikelihood(availability,iterator):
    terms = {}
    for i,a in availability.items() :
        terms[i] = ( a!=0 )

    total = bioMultSum(terms)
    nl = -Sum(log(total),iterator)
    BIOGEME_OBJECT.STATISTICS['Null loglikelihood'] = nl
    return nl

## Computes the number of times each alternative is chosen in the data set and ask Biogeme to include it in the output file..
# \ingroup stats
# \param choiceSet list containing the alternatives for which statistics must be computed.
# \param choice expression producing the id of the chosen alternative.
# \param iterator An iterator on the data file.
# \return A <a href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries" target="_blank">dictionary</a> n with an entry n[i] for each alternative i containing the number of times it is chosen.
# \note Note that availability is ignored here. 
def choiceStatistics(choiceSet,choice,iterator):
    n = {}
    for i in choiceSet:
        n[i] = Sum(choice == i,iterator)
    for i in choiceSet:
        s = 'Alt. %d chosen' % (i)
        BIOGEME_OBJECT.STATISTICS[s] = n[i]
    return n

## Computes the number of times each alternative is declared available in the data set and ask Biogeme to include it in the output file..
# \ingroup stats
# \param availability <a href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries" target="_blank">Dictionary</a> containing for each alternative the expression for its availability. 
# \param iterator An iterator on the data file.
# \return A <a href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries" target="_blank">dictionary</a> n with an entry n[i] for each alternative i containing the number of times it is available.
def availabilityStatistics(availability,iterator):
    n = {}
    for i,a in availability.items():
        n[i] = Sum((a != 0),iterator)
    for i,a in availability.items():
        s = 'Alt. %d available' % (i)
        BIOGEME_OBJECT.STATISTICS[s] = n[i]
    return n    

## Computes the constant loglikelihood from the sample and ask Biogeme to include it in the output file. It assumes that the full choice set is available for each observation.
# \ingroup stats
# \param choiceSet list containing the alternatives in the choice set.
# \param choice expression producing the id of the chosen alternative.
# \param iterator An iterator on the data file.
# \return log likelihood of a logit model where the only parameters are the alternative specific constants. If \f$n_i\f$ is the number of times alternative \f$i\f$ is chosen, then it is given by \f[ \mathcal{L} = \sum_i n_i \ln(n_i) - n \ln(n)  \f] where \f$ n = \sum_i n_i \f$ is the total number of observations.
# \note Note that availability is ignored here. 
def cteLoglikelihood(choiceSet,choice,iterator):
    n = choiceStatistics(choiceSet,choice,iterator)
    terms_l = {}
    terms_tot = {}
    for i in n:
        terms_l[i] = n[i] * log(n[i])
        terms_tot[i] = n[i]
    total = bioMultSum(terms_tot)
    l = bioMultSum(terms_l)
    l -= total * log(total)

    BIOGEME_OBJECT.STATISTICS['Cte loglikelihood (only for full choice sets)'] = l
    return l

