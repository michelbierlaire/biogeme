//-*-c++-*------------------------------------------------------------
//
// File name : bioExprOr.cc
// @date   Sun Apr 29 12:03:56 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprOr.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprOr::bioExprOr(bioExpression* l, bioExpression* r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprOr::~bioExprOr() {

}

  
bioDerivatives* bioExprOr::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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
      throw(bioExceptions(__FILE__,__LINE__,str.str())) ;
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
  if (l->f != 0.0) {
    theDerivatives->f = 1.0 ;
  }
  else { 
    bioDerivatives* r = right->getValueAndDerivatives(literalIds,false,false) ;
    if (r->f != 0.0) {
      theDerivatives->f = 1.0 ;
    }
    else {
      theDerivatives->f = 0.0 ;
    }
  }
  return theDerivatives ;
}

bioString bioExprOr::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "|(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "|" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
