//-*-c++-*------------------------------------------------------------
//
// File name : patMultivariateNormal.h
// Author :    Michel Bierlaire
// Date :      Sat Sep 10 17:14:59 2011
//
//--------------------------------------------------------------------

#ifndef patMultivariateNormal_h
#define patMultivariateNormal_h

#include "patError.h"

class patHybridMatrix ;
class patRandomNumberGenerator ;
#include "patVariables.h"

/**
   @doc Generate pseudo-random vectors from a multivariate normal distribution N(mu,Sigma). 
   @author Michel Bierlaire, EPFL (Sat Sep 10 17:14:59 2011)
 */

class patMultivariateNormal {

 public:
  patMultivariateNormal(patVariables* mu, 
			patHybridMatrix* sigma, 
			patRandomNumberGenerator* normalGenerator,
			patError*& err) ;
  patVariables getNextDraw(patError*& err) ;
  
 private:
  patRandomNumberGenerator* theNormalGenerator ;
  patVariables* theMean ;
  patHybridMatrix* theVarCovar ;
};

#endif
