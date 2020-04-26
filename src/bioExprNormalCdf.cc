//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNormalCdf.cc
// @date   Wed May 30 16:18:10 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprNormalCdf.h"
#include <sstream>
#include <cmath>
#include "bioDebug.h"

bioExprNormalCdf::bioExprNormalCdf(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}
bioExprNormalCdf::~bioExprNormalCdf() {

}

bioDerivatives* bioExprNormalCdf::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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


  bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  theDerivatives->f = theNormalCdf.compute(childResult->f) ;

  if (gradient) {
    bioUInt n = literalIds.size() ;
    bioReal thePdf = invSqrtTwoPi * exp(- childResult->f * childResult->f / 2.0) ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      if (childResult->g[i] == 0.0) {
	theDerivatives->g[i] = 0.0 ;
      }
      else {
	theDerivatives->g[i] = thePdf * childResult->g[i] ;
      }
      if (hessian) {
	for (bioUInt j = i ; j < n ; ++j) {
	  if (childResult->h[i][j] != 0.0) {
	    theDerivatives->h[i][j] = thePdf * childResult->h[i][j] ;
	  }
	  else {
	    theDerivatives->h[i][j] = 0.0 ;
	  }
	  if (childResult->f != 0.0 && 
	      childResult->g[i] != 0 && 
	      childResult->g[j] != 0) {
	    theDerivatives->h[i][j] -= thePdf * childResult->f * childResult->g[i] * childResult->g[j] ;
	  }
	}
      }
    }
  }
  if (hessian) {
    bioUInt n = literalIds.size() ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	theDerivatives->h[j][i] = theDerivatives->h[i][j] ;
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprNormalCdf::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "bioNormalCdf(" << child->print(hp) << ")";
  return str.str() ;

}
