//-*-c++-*------------------------------------------------------------
//
// File name : patBetaLikeParameter.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:26:57 2001
//
//--------------------------------------------------------------------

#ifndef patBetaLikeParameter_h
#define patBetaLikeParameter_h

#include "patString.h"
#include "patType.h"


/**
   @doc Definition for the beta, Box-Cox, Box-Tukey, Mu, scale, NL nests, CNL nest parameters.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 16:26:57 2001)
*/

struct patBetaLikeParameter {
  /**
   */
  patBetaLikeParameter() ;
  /**
   */
  patString name ;
  /**
   */
  patReal defaultValue ;
  /**
   */
  patReal lowerBound ;
  /**
   */
  patReal upperBound ;
  /**
   */
  patBoolean isFixed ;
  /**
   */
  patBoolean hasDiscreteDistribution ;
  /**
   */
  patReal estimated ;
  /**
     index in the vector of unknown parameters to be estimated
  */
  unsigned long index ;
  /**
     IDs are unique and consecutive within each set of parameters
  */
  unsigned long id ;
  
};

ostream& operator<<(ostream &str, const patBetaLikeParameter& x) ;


#endif
