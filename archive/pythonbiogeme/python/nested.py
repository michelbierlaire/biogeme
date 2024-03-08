## \file
# Functions for the nested logit model

from biogeme import *
from mev import *


## Implements the MEV generating function for the nested logit model
# @ingroup models
# @param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @return A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{1}{\mu_m}-1}
# \f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def getMevForNested(V, availability, nests):

    y = {}
    for i, v in V.items():
        y[i] = exp(v)

    Gi = {}
    for m in nests:
        sumdict = []
        for i in m[1]:
            sumdict.append(Elem({0: 0.0, 1: y[i] ** m[0]}, availability[i] != 0))
        sum = bioMultSum(sumdict)
        for i in m[1]:
            Gi[i] = Elem(
                {0: 0, 1: y[i] ** (m[0] - 1.0) * sum ** (1.0 / m[0] - 1.0)},
                availability[i] != 0,
            )
    return Gi


## Implements the nested logit model as a MEV model.
# @ingroup models
# @param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @return Choice probability for the nested logit model, based on the
# derivatives of the MEV generating function produced by the function
# nested::getMevForNested
#
def nested(V, availability, nests, choice):
    Gi = getMevForNested(V, availability, nests)
    P = mev(V, Gi, availability, choice)
    return P


## Implements the log of a nested logit model as a MEV model.
# @ingroup models
# @param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @return Log of choice probability for the nested logit model, based on the
# derivatives of the MEV generating function produced by the function
# nested::getMevForNested
#
def lognested(V, availability, nests, choice):
    Gi = getMevForNested(V, availability, nests)
    logP = logmev(V, Gi, availability, choice)
    return logP


## Implements the nested logit model as a MEV model, where mu is also a
## parameter, if the user wants to test different normalization
## schemes.
# @ingroup models
# @param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @param mu expression producing the value of the top-level scale parameter.
# @return The nested logit choice probability based on the following derivatives of the MEV generating function:
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = \mu e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{\mu}{\mu_m}-1}
# \f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def nestedMevMu(V, availability, nests, choice, mu):

    y = {}
    for i, v in V.items():
        y[i] = exp(v)

    Gi = {}
    for m in nests:
        sum = {}
        for i in m[1]:
            sum[i] = Elem({0: 0, 1: y[i] ** m[0]}, availability[i] != 0)
        for i in m[1]:
            Gi[i] = Elem(
                {
                    0: 0,
                    1: mu * y[i] ** (m[0] - 1.0) * bioMultSum(sum) ** (mu / m[0] - 1.0),
                },
                availability[i] != 0,
            )
    P = mev(V, Gi, availability, choice)
    return P


## Implements the log of the nested logit model as a MEV model, where mu is also a
## parameter, if the user wants to test different normalization
## schemes.
# @ingroup models
# @param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a href="http://docs.python.org/py3k/tutorial/introduction.html#lists">list</a>
#   containing the list of identifiers of the alternatives belonging to
#   the nest.
# Example:
# @code
#  nesta = MUA , [1,2,3]
#  nestb = MUB , [4,5,6]
#  nests = nesta, nestb
# @endcode
# @param choice expression producing the id of the chosen alternative.
# @param mu expression producing the value of the top-level scale parameter.
# @return The nested logit choice probability based on the following derivatives of the MEV generating function:
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J}) = \mu e^{(\mu_m-1)V_i} \left(\sum_{i=1}^{J_m} e^{\mu_m V_i}\right)^{\frac{\mu}{\mu_m}-1}
# \f]
# where \f$m\f$ is the (only) nest containing alternative \f$i\f$, and
# \f$G\f$ is the MEV generating function.
#
def lognestedMevMu(V, availability, nests, choice, mu):

    y = {}
    for i, v in V.items():
        y[i] = exp(v)

    Gi = {}
    for m in nests:
        sum = {}
        for i in m[1]:
            sum[i] = Elem({0: 0, 1: y[i] ** m[0]}, availability[i] != 0)
        for i in m[1]:
            Gi[i] = Elem(
                {
                    0: 0,
                    1: mu * y[i] ** (m[0] - 1.0) * bioMultSum(sum) ** (mu / m[0] - 1.0),
                },
                availability[i] != 0,
            )
    logP = logmev(V, Gi, availability, choice)
    return logP
