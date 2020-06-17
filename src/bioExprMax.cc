//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMax.cc
// @date   Mon Oct 15 15:38:20 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprMax.h"
#include <sstream>
#include "bioSmartPointer.h"
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprMax::bioExprMax(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprMax::~bioExprMax() {
}

  
bioSmartPointer<bioDerivatives>
bioExprMax::getValueAndDerivatives(std::vector<bioUInt> literalIds,
				   bioBoolean gradient,
				   bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  if (theDerivatives == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"theDerivatives") ;
  }
  if (left == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"left") ;
  }
  if (right == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"right") ;
  }
  
  bioSmartPointer<bioDerivatives> leftResult = left->getValueAndDerivatives(literalIds, gradient, hessian) ;
  bioSmartPointer<bioDerivatives> rightResult = right->getValueAndDerivatives(literalIds, gradient, hessian) ;
  if (leftResult->f > rightResult->f) {
    theDerivatives->f = leftResult->f ;
    if (gradient) {
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives->g[k] = leftResult->g[k] ;
	if (hessian) {
	  for (std::size_t l = 0 ; l < literalIds.size() ; ++l) {
	    theDerivatives->h[k][l] = leftResult->h[k][l] ;
	  }
	}
      }
    }
  }
  else { 
    theDerivatives->f = rightResult->f ;
    if (gradient) {
      if (leftResult->f == rightResult->f) {
	if (containsLiterals(literalIds)) {
	  std::cout << "Warning: expression " << print()
		    << " is not differentiable at " << leftResult->f
		    << ". Right derivative is used." << std::endl ;
	}
      }
      for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
	theDerivatives->g[k] = rightResult->g[k] ;
	if (hessian) {
	  for (std::size_t l = 0 ; l < literalIds.size() ; ++l) {
	    theDerivatives->h[k][l] = rightResult->h[k][l] ;
	  }
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprMax::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "bioMax(" << left->print(hp) << "," << right->print(hp) << ")" ;
  return str.str() ;
}


