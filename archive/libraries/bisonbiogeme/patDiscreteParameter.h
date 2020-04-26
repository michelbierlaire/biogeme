//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteParameter.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed Dec  1 10:36:33 2004
//
//--------------------------------------------------------------------

#ifndef patDiscreteParameter_h
#define patDiscreteParameter_h

#include "patType.h"

struct patBetaLikeParameter ;
class patArithRandom ;
/**
 */
struct patDiscreteTerm {
  patBetaLikeParameter* massPoint ;
  patBoolean random ;
  patArithRandom* massPointRandom ;
  patBetaLikeParameter* probability ;
} ;

/**
   @doc Definition of the structure of a random parameter with a discrete distribution, that is the list of mass point with associate probability. 
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Wed Dec  1 10:36:33 2004)
 */

class patDiscreteParameter {
public:
  /**
   */
  patString name ;

  /**
     Pointer to the beta-like parameter capturing the discrete parameter
   */
  patBetaLikeParameter* theParameter ; 

  /**
   */
  vector<patDiscreteTerm> listOfTerms ;

};
#endif
