//-*-c++-*------------------------------------------------------------
//
// File name : patRandomParameter.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar  4 08:32:08 2003
//
//--------------------------------------------------------------------

#ifndef patRandomParameter_h
#define patRandomParameter_h

#include <vector>
#include "patString.h"
#include "patType.h"
#include "patDistribType.h"

struct patBetaLikeParameter ;



/**
   @doc Definition of the structure of a random parameter, that is its
   location parameter, scale parameter and, when relevant,  covariances. 
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Mar  4 08:32:08 2003)
 */

struct patRandomParameter {

  /**
   */
  patRandomParameter() ;

  /**
   */
  patString name ;

  /**
     Type of the distribution
   */
  patDistribType type ;

  /**
   */
  patBetaLikeParameter* location ;

  /**
   */
  patBetaLikeParameter* scale ;

  /**
     The correlation structure is defined as a lower triangular matrix. It
     means that, if $\beta_a$ is correlated with $\beta_b$, $\beta_b$
     will appear in the list of correlated parameters of $\beta_a$,
     but not the other way around.
   */
  vector<patRandomParameter*> correlatedParameters ;

  /**
     Index in the vector of draws in the patIndividualData object.
   */
  unsigned long index ;

  /**
     TRUE is the random parameter is used for panel data, and is
     individual specific.
   */
  patBoolean panel ;
  
  /**
     Mass at zero
   */
  patReal massAtZero ;

};

ostream& operator<<(ostream &str, const patRandomParameter& x) ;

patBoolean operator<(const patRandomParameter& p1, 
		     const patRandomParameter& p2) ;

#endif
