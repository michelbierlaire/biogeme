//-*-c++-*------------------------------------------------------------
//
// File name : bioExprUnaryMinus.cc
// @date   Fri Apr 13 11:38:02 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprUnaryMinus.h"
#include "bioDebug.h"
#include <sstream>
#include "bioSmartPointer.h"

bioExprUnaryMinus::bioExprUnaryMinus(bioSmartPointer<bioExpression>  c) :
  child(c) {
  listOfChildren.push_back(c) ;
}
bioExprUnaryMinus::~bioExprUnaryMinus() {
}

bioSmartPointer<bioDerivatives> bioExprUnaryMinus::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  bioUInt n = literalIds.size() ;
  bioSmartPointer<bioDerivatives> childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  theDerivatives->f = - childResult->f ;
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] = - childResult->g[i] ;
      if (hessian) {
	for (bioUInt j = 0 ; j < n ; ++j) {
	  theDerivatives->h[i][j] = - childResult->h[i][j] ;
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprUnaryMinus::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "-" << child->print(hp) ;
  return str.str() ;

}

