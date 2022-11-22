//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLess.cc
// @date   Thu Apr 19 07:26:05 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprLess.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprLess::bioExprLess(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}

bioExprLess::~bioExprLess() {

}

const bioDerivatives* bioExprLess::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {
  
  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  theDerivatives.resize(literalIds.size()) ;

  if (gradient) {
    if (containsLiterals(literalIds)) {
      std::stringstream str ;
      str << "Expression Less is not differentiable" << std::endl ; 
      throw(bioExceptions(__FILE__,__LINE__,str.str())) ;
    }
    theDerivatives.setDerivativesToZero() ;
  }

  if (left->getValue() < right->getValue()) {
    theDerivatives.f = 1.0 ;
  }
  else {
    theDerivatives.f = 0.0 ;
  }
  
  return &theDerivatives ;
}

bioString bioExprLess::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "<(" << left->print(hp) << "<" << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "<" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
