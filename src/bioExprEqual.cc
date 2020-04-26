//-*-c++-*------------------------------------------------------------
//
// File name : bioExprEqual.cc
// @date   Thu Apr 19 07:16:34 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprEqual.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprEqual::bioExprEqual(bioExpression* l, bioExpression* r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprEqual::~bioExprEqual() {

}

  
bioDerivatives* bioExprEqual::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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
      str << "Expression Equal is not differentiable" << std::endl ; 
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
  bioDerivatives* r = right->getValueAndDerivatives(literalIds,false,false) ;
  if (l->f == r->f) {
    theDerivatives->f = 1.0 ;
  }
  else {
    theDerivatives->f = 0.0 ;
  }
  return theDerivatives ;
}

bioString bioExprEqual::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "==(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "==" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
