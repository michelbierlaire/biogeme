## @file
# Functions for the MEV model

from biogeme import *

## Choice probability for a MEV model.
# @ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function. If an alternative \f$i\f$ is not available, then \f$G_i = 0\f$.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param choice Expression producing the id of the chosen alternative.
# @return Choice probability of the MEV model, given by
#  \f[
#    \frac{e^{V_i + \ln G_i(e^{V_1},\ldots,e^{V_J})}}{\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}}
#  \f]
#
# \code
# def mev(V,Gi,av,choice) :
#     H = {}
#     for i,v in V.items() :
#        H[i] =  Elem({0:0, 1: v + log(Gi[i])},Gi[i]!=0)  
#     P = bioLogit(H,av,choice)
#     return P
# \endcode
def mev(V,Gi,av,choice) :
    H = {}
    for i,v in V.items() :
        H[i] =  Elem({0:0, 1: v + log(Gi[i])},av[i]!=0)  
    P = bioLogit(H,av,choice)
    return P

## Log of the choice probability for a MEV model.
# @ingroup models
# \param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function. If an alternative \f$i\f$ is not available, then \f$G_i = 0\f$.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param choice Expression producing the id of the chosen alternative.
# @return Log of the choice probability of the MEV model, given by
#  \f[
#    V_i + \ln G_i(e^{V_1},\ldots,e^{V_J}) - \log\left(\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}\right)
#  \f]
#
# \code
# def logmev(V,Gi,av,choice) :
#     H = {}
#     for i,v in V.items() :
#        H[i] =  Elem({0:0, 1: v + log(Gi[i])},Gi[i]!=0)  
#     P = bioLogLogit(H,av,choice)
#     return P
# \endcode
def logmev(V,Gi,av,choice) :
    H = {}
    for i,v in V.items() :
        H[i] =  Elem({0:0, 1: v + log(Gi[i])},av[i]!=0)  
    logP = bioLogLogit(H,av,choice)
    return logP



## Choice probability for a MEV model, including the correction for endogenous sampling as proposed by <a href="http://dx.doi.org/10.1016/j.trb.2007.09.003" taret="_blank">Bierlaire, Bolduc and McFadden (2008)</a>.
# @ingroup biogeme
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param correction A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the correction. Typically, it is a value, or a
# parameter to be estimated.
# @param choice Expression producing the id of the chosen alternative.
# @return Choice probability of the MEV model, given by
#  \f[
#    \frac{e^{V_i + \ln G_i(e^{V_1},\ldots,e^{V_J})}}{\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}}
#  \f]
#
# \code
# def mev_selectionBias(V,Gi,av,correction,choice) :
#     H = {}
#     for i,v in V.items() :
#         H[i] = v + log(Gi[i]) + correction[i]
#     P = bioLogit(H,av,choice)
#     return P
# \endcode
def mev_selectionBias(V,Gi,av,correction,choice) :
    H = {}
    for i,v in V.items() :
        H[i] = v + log(Gi[i]) + correction[i]

    P = bioLogit(H,av,choice)
            
    return P



## Log of choice probability for a MEV model, including the correction for endogenous sampling as proposed by <a href="http://dx.doi.org/10.1016/j.trb.2007.09.003" taret="_blank">Bierlaire, Bolduc and McFadden (2008)</a>.
# @ingroup biogeme
# @ingroup models
# @param V A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the utility function.
# @param Gi A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the function
# \f[
#   \frac{\partial G}{\partial y_i}(e^{V_1},\ldots,e^{V_J})
#\f]
# where \f$G\f$ is the MEV generating function.
# @param av A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with its
# availability condition.
# @param correction A <a
# href="http://docs.python.org/py3k/tutorial/datastructures.html#dictionaries"
# target="_blank">dictionary</a> mapping each alternative id with the
# expression of the correction. Typically, it is a value, or a
# parameter to be estimated.
# @param choice Expression producing the id of the chosen alternative.
# @return Log of choice probability of the MEV model, given by
#  \f[
#    V_i + \ln G_i(e^{V_1},\ldots,e^{V_J}) - \log\left(\sum_j e^{V_j + \ln G_j(e^{V_1},\ldots,e^{V_J})}\right)
#  \f]
#
# \code
# def logmev_selectionBias(V,Gi,av,correction,choice) :
#     H = {}
#     for i,v in V.items() :
#         H[i] = v + log(Gi[i]) + correction[i]
#     P = bioLogLogit(H,av,choice)
#     return P
# \endcode
def logmev_selectionBias(V,Gi,av,correction,choice) :
    H = {}
    for i,v in V.items() :
        H[i] = v + log(Gi[i]) + correction[i]

    P = bioLogLogit(H,av,choice)
            
    return P




    
