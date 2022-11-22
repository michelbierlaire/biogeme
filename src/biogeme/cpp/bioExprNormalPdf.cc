//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNormalPdf.cc
// @date   Tue Aug 20 08:22:05 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprNormalPdf.h"
#include <sstream>
#include <cmath>
#include "bioDebug.h"

bioExprNormalPdf::bioExprNormalPdf(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}
bioExprNormalPdf::~bioExprNormalPdf() {

}

const bioDerivatives* bioExprNormalPdf::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;
  theDerivatives.resize(literalIds.size()) ;

  const bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioReal x = - childResult->f * childResult->f / 2.0 ;
  if (x <= bioLogMaxReal::the()) { 
    theDerivatives.f = exp(x) * 0.3989422804 ;
  }
  else {
    theDerivatives.f = std::numeric_limits<bioReal>::max() ;
  }

  
  if (gradient) {
    bioUInt n = literalIds.size() ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      if (childResult->g[i] == 0.0) {
	theDerivatives.g[i] = 0.0 ;
      }
      else {
	theDerivatives.g[i] = - x * theDerivatives.f * childResult->g[i] ;
      }
      if (hessian) {
	for (bioUInt j = i ; j < n ; ++j) {
	  if (childResult->h[i][j] != 0.0) {
	    theDerivatives.h[i][j] = TO BE IMPLEMENTED (AND TESTED)
	  }
	  else {
	    theDerivatives.h[i][j] = 0.0 ;
	  }
	  if (childResult->f != 0.0 && 
	      childResult->g[i] != 0 && 
	      childResult->g[j] != 0) {
	    theDerivatives.h[i][j] -= thePdf * childResult->f * childResult->g[i] * childResult->g[j] ;
	  }
	}
      }
    }
  }
  if (hessian) {
    bioUInt n = literalIds.size() ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	theDerivatives.h[j][i] = theDerivatives.h[i][j] ;
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprNormalPdf::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "bioNormalPdf(" << child->print(hp) << ")";
  return str.str() ;

}
