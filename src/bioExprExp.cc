//-*-c++-*------------------------------------------------------------
//
// File name : bioExprExp.cc
// @date   Tue Apr 17 12:13:00 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprExp.h"
#include "bioExceptions.h"
#include "bioDebug.h"
#include "bioSmartPointer.h"
#include <cmath>
#include <sstream> 

bioExprExp::bioExprExp(bioSmartPointer<bioExpression>  c) :
  child(c) {

  listOfChildren.push_back(c) ;
}

bioExprExp::~bioExprExp() {
}

bioSmartPointer<bioDerivatives>
bioExprExp::getValueAndDerivatives(std::vector<bioUInt> literalIds,
				   bioBoolean gradient,
				   bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  bioUInt n = literalIds.size() ;
  bioSmartPointer<bioDerivatives> childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  if (childResult->f <= bioLogMaxReal::the()) { 
    theDerivatives->f = exp(childResult->f) ;
  }
  else {
    theDerivatives->f = std::numeric_limits<bioReal>::max() ;
  }
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] = theDerivatives->f * childResult->g[i] ;
      if (hessian) {
	for (bioUInt j = 0 ; j < n ; ++j) {
	  theDerivatives->h[i][j] = theDerivatives->f * (childResult->h[i][j] +  childResult->g[i] *  childResult->g[j]);
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprExp::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "exp(" << child->print(hp) << ")";
  return str.str() ;

}
