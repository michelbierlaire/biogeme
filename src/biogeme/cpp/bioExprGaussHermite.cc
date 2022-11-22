//-*-c++-*------------------------------------------------------------
//
// File name : bioExprGaussHermite.cc
// Author :    Michel Bierlaire
// Date :      Sat May 21 19:25:23 2011
// Modified for biogemepython 3.0: Wed May  9 17:51:31 2018

//
//--------------------------------------------------------------------


#include "bioExprGaussHermite.h"
#include "bioExpression.h"
#include "bioDebug.h"

bioExprGaussHermite::bioExprGaussHermite(bioExpression* e,
					 std::vector<bioUInt> derivl,
					 bioUInt l,
					 bioBoolean wg,
					 bioBoolean wh) : 
  withGradient(wg),
  withHessian(wh),
  theExpression(e),
  derivLiteralIds(derivl),
  rvId(l) {

  theExpression->setRandomVariableValuePtr(rvId,&theValue) ;


  
}

std::vector<bioReal> bioExprGaussHermite::getValue(bioReal x) {
  std::vector<bioReal> result ;
  theValue = x ;
  bioUInt n = derivLiteralIds.size() ;
  const bioDerivatives* fgh = theExpression->getValueAndDerivatives(derivLiteralIds,withGradient,withHessian) ;
  result.push_back(fgh->f) ;
  if (withGradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      result.push_back(fgh->g[i]) ;
    }
    if (withHessian) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	for (bioUInt j = i ; j < n ; ++j) {
	  result.push_back(fgh->h[i][j]) ;
	}
      }
    }
  }
  return result ;
}

bioUInt bioExprGaussHermite::getSize() const {
  if (withGradient) {
      bioUInt n = derivLiteralIds.size() ; 
    if (withHessian) {
      return 1 + n + (n * (n+1) / 2) ;
    }
    else {
      return 1 + n ;
    }
  }
  else {
    return 1 ;
  }
}


