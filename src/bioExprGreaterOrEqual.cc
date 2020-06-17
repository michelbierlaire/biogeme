//-*-c++-*------------------------------------------------------------
//
// File name : bioExprGreaterOrEqual.cc
// @date   Thu Apr 19 07:24:15 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprGreaterOrEqual.h"
#include <sstream>
#include "bioSmartPointer.h"
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprGreaterOrEqual::bioExprGreaterOrEqual(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) :
  left(l), right(r) {

  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;
}

bioExprGreaterOrEqual::~bioExprGreaterOrEqual() {
}

bioSmartPointer<bioDerivatives>
bioExprGreaterOrEqual::getValueAndDerivatives(std::vector<bioUInt> literalIds,
					      bioBoolean gradient,
					      bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  if (gradient) {
    if (containsLiterals(literalIds)) {
      std::stringstream str ;
      str << "Expression GreaterOrEqual is not differentiable" << std::endl ; 
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    if (hessian) {
      theDerivatives->setDerivativesToZero() ;
    }
    else {
      theDerivatives->setGradientToZero() ;
    }
  }
  

  if (left->getValue() >= right->getValue()) {
    theDerivatives->f = 1.0 ;
  }
  else {
    theDerivatives->f = 0.0 ;
  }
  
  return theDerivatives ;
}

bioString bioExprGreaterOrEqual::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << ">=(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << ">=" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
