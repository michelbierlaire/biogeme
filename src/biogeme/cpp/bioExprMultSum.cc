//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMultSum.cc
// @date   Wed Apr 18 11:05:06 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <sstream>
#include "bioDebug.h"
#include "bioExprMultSum.h"
#include "bioExceptions.h"

bioExprMultSum::bioExprMultSum(std::vector<bioExpression*> e) :
  expressions(e) {
  listOfChildren = e ;
}

bioExprMultSum::~bioExprMultSum() {

}

const bioDerivatives* bioExprMultSum::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							     bioBoolean gradient,
							     bioBoolean hessian) {

  //DEBUG_MESSAGE("bioExprMultSum getValueAndDerivatives") ;
  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;
  
  theDerivatives.resize(literalIds.size()) ;

  theDerivatives.f = 0.0 ;
  theDerivatives.setDerivativesToZero() ;
  for (std::vector<bioExpression*>::iterator i = expressions.begin();
       i != expressions.end() ;
       ++i) {
    const bioDerivatives* fgh = (*i)->getValueAndDerivatives(literalIds,gradient,hessian) ;
    
    theDerivatives.f += fgh->f ;
    if (gradient) {
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives.g[k] += fgh->g[k] ;
	if (hessian) {
	  for (std::size_t l = k ; l < literalIds.size() ; ++l) {
	    theDerivatives.h[k][l] += fgh->h[k][l] ;
	  }
	}
      }
    }
  }
  if (hessian) {
    // Fill the symmetric part of the matrix
    for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
      for (std::size_t l = k+1 ; l < literalIds.size() ; ++l) {
	theDerivatives.h[l][k]  = theDerivatives.h[k][l] ;
      }
    }
  }
  //  DEBUG_MESSAGE("MultiSum calculated " << str.length()) ;
  //DEBUG_MESSAGE("bioExprMultSum getValueAndDerivatives: RETURN") ;
  return &theDerivatives ;
}

bioString bioExprMultSum::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "MultiSum(" ;
    for (std::vector<bioExpression*>::const_iterator i = expressions.begin() ;
	 i != expressions.end() ;
	 ++i) {
      if (i != expressions.begin()) {
	str << " , " ;
      }
      str << (*i)->print(hp) ;
    }
    str << ")" ;
  }
  else {
    str << "(" ;
    for (std::vector<bioExpression*>::const_iterator i = expressions.begin() ;
	 i != expressions.end() ;
	 ++i) {
      if (i != expressions.begin()) {
	str << " + " ;
      }
      str << (*i)->print(hp) ;
    }
    str << ")" ;
  }
  return str.str() ;
}
