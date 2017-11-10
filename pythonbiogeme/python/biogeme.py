"""
File name : biogeme.py
Author :    Michel Bierlaire and Mamy Fetiarison
Date :      Fri May 15 14:23:51 2009
"""
from bio_expression import *
from bio_iterator import *
from bioMatrix import *

##
## Class used for generating STAN files (see http://mc-stan.org)
class STAN:
   CATEGORICAL = {}
   NORMAL = {}
   GAMMA = {}

## This class gathers the components needed by biogeme to perform the
## estimation or the simulation.
class BIOGEME_OBJECT:
   ## Expression of the likelihood function to be maximized. Example: @code BIOGEME_OBJECT.ESTIMATE = Sum(log(prob),'obsIter') @endcode
   ESTIMATE = None
   ## Enumerate expression to perform sample enumeration. Example: 
   # @code 
   # simulate = Enumerate('Choice prob.', prob)
   # BIOGEME_OBJECT.SIMULATE = Enumerate(simulate,'obsIter') 
   # @endcode
   SIMULATE = None
   ## Expression identifying the observations to be ignored in the data file.
   # Example:
   # @code
   # BIOGEME_OBJECT.EXCLUDE = (TRAVEL_TIME < 0)
   # @endcode
   EXCLUDE = None
   ## Expression computing the weight of each observation, for
   ## WESML. Typically, a variable in the data file.
   # Example: @code BIOGEME_OBJECT.WEIGHT = weight @endcode
   WEIGHT = None
   ## Object of type bioMatrix that contains the variance covariance
   ## matrix of the estimators, in order to perform a sensitivity
   ## analysis of the simulated quantities
   VARCOVAR = None
   ## Expression describing a simulation method for the Bayesian
   ## estimation of the parameters.
   BAYESIAN = None
   ## Boolean indicator for the generation of STAN files
   STAN = STAN()
   ## Dictionary of expressions to be reported in the output file
   FORMULAS = {}

   ## Dictionary of expressions computing statistics on the data.
   # The index of each entry is associated with the value in the report. Example calculating the null loglikelihood of a choice model:
   # @code
   #    total = 0 
   #    for i,a in availability.items() :
   #        total += (a != 0)
   #    nl = -Sum(log(total),iterator)
   #    BIOGEME_OBJECT.STATISTICS['Null loglikelihood'] = nl
   # @endcode
   STATISTICS = {}
   ## Dictionary of expressions defining constraints on the parameters
   # to be estimated. In general, it is preferable not to use these
   # constraints. There is most of the time a way to write the model
   # with unconstrained parameters. The two typical examples are:
   # - Constrain a parameter beta to be nonnegative: replace it by the exponential of another parameter.
   # - A set of parameters b1, b2, ... bn must sum up to one. Replace
   #  them by their normalized version: bi = ci / (c1 + c2 + ... +
   #  cn), where c1, ... cn are unconstrained parameters to be
   #  estimated.
   # - A combination of the two gives bi = exp(ci) / (exp(c1) + exp(c2) + ... +
   #  exp(cn))
   CONSTRAINTS = {}
   ## Dictionary of the various draws used for Monte-Carlo simulation.
   DRAWS = {}

   ## Used to set the value of the parameters. The syntax is 
   # @code BIOGEME_OBJECT.PARAMETERS['parameterName'] = 'value' @endcode
   PARAMETERS = {}
   
   def __init__(self) :
      # Only one object can be instantiate
      BIOGEME_OBJECT.__init__ = BIOGEME_OBJECT.__once__

   def __once__(self):
      print("No further BIOGEME_OBJECT instance can be created!")
      return


"""
Numeric constants
"""
one = Numeric(1)
zero = Numeric(0)

