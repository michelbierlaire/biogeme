## \file
# Functions for the cross-nested logit model

from biogeme import *
from mev import *


## Implements the cross-nested logit model as a MEV model.
# \ingroup models
# \param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest.
# Example with two nests and 6 alternatives:
# \code
# alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
# alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
# nesta = MUA , alphaA
# nestb = MUB , alphaB
# nests = nesta, nestb
# \endcode
# \return Choice probability for the cross-nested logit model.
#
def cnl_avail(V, availability, nests, choice):
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = []
    for m in nests:
        biosumlist = []
        for i, a in m[1].items():
            biosumlist.append(
                Elem({0: 0, 1: a ** (m[0]) * exp(m[0] * (V[i]))}, availability[i] != 0)
            )
        biosum = bioMultSum(biosumlist)
        for i, a in m[1].items():
            Gidict[i].append(
                Elem(
                    {
                        0: 0,
                        1: (biosum ** ((1.0 / m[0]) - 1.0))
                        * (a ** m[0])
                        * exp((m[0] - 1.0) * (V[i])),
                    },
                    availability[i] != 0,
                )
            )
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    P = mev(V, Gi, availability, choice)
    return P


## Implements the log of the cross-nested logit model as a MEV model.
# \ingroup models
# \param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest.
# Example with two nests and 6 alternatives:
# \code
# alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
# alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
# nesta = MUA , alphaA
# nestb = MUB , alphaB
# nests = nesta, nestb
# \endcode
# \return Choice probability for the cross-nested logit model.
#
def logcnl_avail(V, availability, nests, choice):
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = []
    for m in nests:
        biosumlist = []
        for i, a in m[1].items():
            biosumlist.append(
                Elem({0: 0, 1: a ** (m[0]) * exp(m[0] * (V[i]))}, availability[i] != 0)
            )
        biosum = bioMultSum(biosumlist)
        for i, a in m[1].items():
            Gidict[i].append(
                Elem(
                    {
                        0: 0,
                        1: (biosum ** ((1.0 / m[0]) - 1.0))
                        * (a ** m[0])
                        * exp((m[0] - 1.0) * (V[i])),
                    },
                    availability[i] != 0,
                )
            )
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    logP = logmev(V, Gi, availability, choice)
    return logP


## Implements the cross-nested logit model as a MEV model with the homogeneity parameters is explicitly involved
# \ingroup models
# \param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest.
# Example with two nests and 6 alternatives:
# \code
# alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
# alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
# nesta = MUA , alphaA
# nestb = MUB , alphaB
# nests = nesta, nestb
# \endcode
# \param bmu Homogeneity parameter \f$\mu\f$.
# \return Choice probability for the cross-nested logit model.
def cnlmu(V, availability, nests, choice, bmu):
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = []
    for m in nests:
        biosumdict = []
        for i, a in m[1].items():
            biosumdict.append(
                Elem(
                    {0: 0, 1: a ** (m[0] / bmu) * exp(m[0] * (V[i]))},
                    availability[i] != 0,
                )
            )
        biosum = bioMultSum(biosumdict)
        for i, a in m[1].items():
            Gidict[i].append(
                Elem(
                    {
                        0: 0,
                        1: bmu
                        * (biosum ** ((bmu / m[0]) - 1.0))
                        * (a ** (m[0] / bmu))
                        * exp((m[0] - 1.0) * (V[i])),
                    },
                    availability[i] != 0,
                )
            )
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    P = mev(V, Gi, availability, choice)
    return P


## Implements the log of the cross-nested logit model as a MEV model with the homogeneity parameters is explicitly involved
# \ingroup models
# \param util A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# \param availability A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# \param nests A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing as many items as nests. Each item is also a <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#tuples-and-sequences">tuple</a>
# containing two items:
# - An <a href="http://biogeme.epfl.ch/expressions.html">expression</a>
#   representing the nest parameter.
# - A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping the alternative ids with the cross-nested parameters for the corresponding nest.
# Example with two nests and 6 alternatives:
# \code
# alphaA = {1: alpha1a,
#          2: alpha2a,
#          3: alpha3a,
#          4: alpha4a,
#          5: alpha5a,
#          6: alpha6a}
# alphaB = {1: alpha1b,
#          2: alpha2b,
#          3: alpha3b,
#          4: alpha4b,
#          5: alpha5b,
#          6: alpha6b}
# nesta = MUA , alphaA
# nestb = MUB , alphaB
# nests = nesta, nestb
# \endcode
# \param bmu Homogeneity parameter \f$\mu\f$.
# \return Log of choice probability for the cross-nested logit model.
def logcnlmu(V, availability, nests, choice, bmu):
    Gi = {}
    Gidict = {}
    for k in V:
        Gidict[k] = []
    for m in nests:
        biosumlist = []
        for i, a in m[1].items():
            biosumlist.append(
                Elem(
                    {0: 0, 1: a ** (m[0] / bmu) * exp(m[0] * (V[i]))},
                    availability[i] != 0,
                )
            )
        biosum = bioMultSum(biosumlist)
        for i, a in m[1].items():
            Gidict[i].append(
                Elem(
                    {
                        0: 0,
                        1: bmu
                        * (biosum ** ((bmu / m[0]) - 1.0))
                        * (a ** (m[0] / bmu))
                        * exp((m[0] - 1.0) * (V[i])),
                    },
                    availability[i] != 0,
                )
            )
    for k in V:
        Gi[k] = bioMultSum(Gidict[k])
    logP = logmev(V, Gi, availability, choice)
    return logP
