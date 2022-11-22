//-*-c++-*------------------------------------------------------------
//
// File name : bioExprSum.cc
// @date   Fri Apr 13 10:27:14 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprSum.h"
#include "bioDebug.h"
#include <sstream>

bioExprSum::bioExprSum(bioExpression* c, std::vector< std::vector<bioReal> >* d) :
  child(c), data(d) {
  listOfChildren.push_back(c) ;

}

bioExprSum::~bioExprSum() {

}
  
const bioDerivatives* bioExprSum::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						   bioBoolean gradient,
						   bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  bioUInt n = literalIds.size() ;
  theDerivatives.resize(n) ;

  theDerivatives.setToZero() ;
  for (std::vector< std::vector<bioReal> >::iterator rowIterator = data->begin() ;
       rowIterator != data->end() ;
       ++rowIterator) {
    child->setVariables(&(*rowIterator)) ;
    const bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
    theDerivatives.f += childResult->f ;
    if (gradient) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	theDerivatives.g[i] += childResult->g[i] ;
	if (hessian) {
	  for (bioUInt j = i ; j < n ; ++j) {
	    theDerivatives.h[i][j] += childResult->h[i][j] ;
	  }
	}
      }
    }
  }
  // Fill the symmetric part of the hessian
  if (hessian) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = 0 ; j < i ; ++j) {
	theDerivatives.h[i][j] = theDerivatives.h[j][i] ;
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprSum::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "Sum(" << child->print(hp) << ")" ;
  return str.str() ;
}

