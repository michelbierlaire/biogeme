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
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprMax::bioExprMax(bioExpression* l, bioExpression* r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprMax::~bioExprMax() {

}

  
bioDerivatives* bioExprMax::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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

  if (gradient) {
    if (containsLiterals(literalIds)) {
      std::stringstream str ;
      str << "Expression "+print()+" is not differentiable" << std::endl ; 
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    if (hessian) {
      theDerivatives->setDerivativesToZero() ;
    }
    else {
      theDerivatives->setGradientToZero() ;
    }
  }

  
  if (theDerivatives == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"theDerivatives") ;
  }
  if (left == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"left") ;
  }
  if (right == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"right") ;
  }
  
  bioDerivatives* l = left->getValueAndDerivatives(literalIds,false,false) ;
  bioDerivatives* r = right->getValueAndDerivatives(literalIds,false,false) ;
  if (l->f >= r->f) {
    theDerivatives->f = l->f ;
  }
  else { 
    theDerivatives->f = r->f ;
  }
  return theDerivatives ;
}

bioString bioExprMax::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "bioMax(" << left->print(hp) << "," << right->print(hp) << ")" ;
  return str.str() ;
}


