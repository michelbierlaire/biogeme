//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMontecarlo.cc
// @date   Tue May  8 10:31:12 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprMontecarlo.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprMontecarlo::bioExprMontecarlo(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}
bioExprMontecarlo::~bioExprMontecarlo() {

}

bioDerivatives* bioExprMontecarlo::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {

  if (theDerivatives == NULL) {
    theDerivatives = new bioDerivatives(literalIds.size()) ;
  }
  else {
    if (gradient && theDerivatives->getSize() != literalIds.size()) {
      delete(theDerivatives) ;
      theDerivatives = new bioDerivatives(literalIds.size()) ;
    }
  }

  theDerivatives->f = 0.0 ;
  if (gradient) {
    if (hessian) {
      theDerivatives->setDerivativesToZero() ;
    }
    else {
      theDerivatives->setGradientToZero() ;
    }
  }

  if (numberOfDraws == 0) {
    throw bioExceptions(__FILE__,__LINE__,"Cannot perform Monte-Carlo integration with no draws.") ;
  }

  bioUInt n = literalIds.size() ;
  child->setDrawIndex(&drawIndex) ;
  for (drawIndex = 0 ; drawIndex < numberOfDraws ; ++drawIndex) {
    bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
    theDerivatives->f += childResult->f ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	theDerivatives->g[i] += childResult->g[i] ;
	if (hessian) {
	  for (bioUInt j = i ; j < n ; ++j) {
	    theDerivatives->h[i][j] += childResult->h[i][j] ;
	  }
	}
      }
    }
  }
  
  theDerivatives->f /= bioReal(numberOfDraws) ;
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] /= bioReal(numberOfDraws) ;
      if (hessian) {
	for (bioUInt j = i ; j < n ; ++j) {
	  theDerivatives->h[i][j] /= bioReal(numberOfDraws) ;
	}
      }
    }
  }
  if (hessian) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	theDerivatives->h[j][i] = theDerivatives->h[i][j] ;
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprMontecarlo::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "Montecarlo(" << child->print(hp) << ")";
  return str.str() ;

}
