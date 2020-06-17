//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMinus.cc
// @date   Fri Apr 13 11:38:02 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprMinus.h"
#include "bioDebug.h"

#include "bioSmartPointer.h"
#include <sstream>

bioExprMinus::bioExprMinus(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}

bioExprMinus::~bioExprMinus() {
}

bioSmartPointer<bioDerivatives>
bioExprMinus::getValueAndDerivatives(std::vector<bioUInt> literalIds,
				     bioBoolean gradient,
				     bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  bioUInt n = literalIds.size() ;
  bioSmartPointer<bioDerivatives> leftResult = left->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioSmartPointer<bioDerivatives> rightResult = right->getValueAndDerivatives(literalIds,gradient,hessian) ;
  theDerivatives->f = leftResult->f - rightResult->f ;
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] = leftResult->g[i] - rightResult->g[i] ;
      if (hessian) {
	for (bioUInt j = 0 ; j < n ; ++j) {
	  theDerivatives->h[i][j] = leftResult->h[i][j] - rightResult->h[i][j] ;
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprMinus::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "-(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "-" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
