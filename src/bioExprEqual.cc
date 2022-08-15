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


const bioDerivatives* bioExprEqual::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							   bioBoolean gradient,
							   bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  theDerivatives.resize(literalIds.size()) ;

  if (gradient) {
    if (containsLiterals(literalIds)) {
      std::stringstream str ;
      str << "Expression Equal is not differentiable" << std::endl ;
      str << "[" << print() << "]" << std::endl ;
      throw(bioExceptions(__FILE__,__LINE__,str.str())) ;
    }
    theDerivatives.setDerivativesToZero() ;
  }

  if (left == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"left") ;
  }
  if (right == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"right") ;
  }

  const bioDerivatives* l = left->getValueAndDerivatives(literalIds,false,false) ;
  const bioDerivatives* r = right->getValueAndDerivatives(literalIds,false,false) ;
  if (l->f == r->f) {
    theDerivatives.f = 1.0 ;
  }
  else {
    theDerivatives.f = 0.0 ;
  }
  return &theDerivatives ;
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
