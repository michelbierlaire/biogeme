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

bioDerivatives* bioExprMultSum::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }

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
  for (std::vector<bioExpression*>::iterator i = expressions.begin();
       i != expressions.end() ;
       ++i) {
    // if (str.length() == 1003) {
    //   DEBUG_MESSAGE("-> " << (*i)->print(true)) ;
    // } 
    bioDerivatives* fgh = (*i)->getValueAndDerivatives(literalIds,gradient,hessian) ;
    // if (str.length() == 1003) {
    //   DEBUG_MESSAGE("-> OK " << fgh->f) ;
    // } 
    theDerivatives->f += fgh->f ;
    if (gradient) {
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives->g[k] += fgh->g[k] ;
	if (hessian) {
	  for (std::size_t l = k ; l < literalIds.size() ; ++l) {
	    theDerivatives->h[k][l] += fgh->h[k][l] ;
	  }
	}
      }
    }
  }
  if (hessian) {
    // Fill the symmetric part of the matrix
    for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
      for (std::size_t l = k+1 ; l < literalIds.size() ; ++l) {
	theDerivatives->h[l][k]  = theDerivatives->h[k][l] ;
      }
    }
  }
  //  DEBUG_MESSAGE("MultiSum calculated " << str.length()) ;
  return theDerivatives ;
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
