//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMin.cc
// @date   Mon Oct 15 15:34:56 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprMin.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprMin::bioExprMin(bioExpression* l, bioExpression* r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprMin::~bioExprMin() {

}

  
const bioDerivatives* bioExprMin::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						     bioBoolean gradient,
						     bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  theDerivatives.resize(literalIds.size()) ;

  if (gradient) {
    if (containsLiterals(literalIds)) {
      std::cout << "Warning: expression " << print()
		<< " is not differentiable everywhere. " << std::endl ;
    }
  }

  if (left == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"left") ;
  }
  if (right == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"right") ;
  }
  
  const bioDerivatives* leftResult = left->getValueAndDerivatives(literalIds, gradient, hessian) ;
  const bioDerivatives* rightResult = right->getValueAndDerivatives(literalIds, gradient, hessian) ;
  if (leftResult->f <= rightResult->f) {
    theDerivatives.f = leftResult->f ;
    if (gradient) {
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives.g[k] = leftResult->g[k] ;
	if (hessian) {
	  for (std::size_t l = 0 ; l < literalIds.size() ; ++l) {
	    theDerivatives.h[k][l] = leftResult->h[k][l] ;
	  }
	}
      }
    }
  }
  else { 
    theDerivatives.f = rightResult->f ;
    if (gradient) {
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives.g[k] = rightResult->g[k] ;
	if (hessian) {
	  for (std::size_t l = 0 ; l < literalIds.size() ; ++l) {
	    theDerivatives.h[k][l] = rightResult->h[k][l] ;
	  }
	}
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprMin::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "bioMin(" << left->print(hp) << "," << right->print(hp) << ")" ;
  return str.str() ;
}
