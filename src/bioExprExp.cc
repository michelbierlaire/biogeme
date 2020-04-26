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
#include <cmath>
#include <sstream> 

bioExprExp::bioExprExp(bioExpression* c) :
  child(c) {

  listOfChildren.push_back(c) ;
}

bioExprExp::~bioExprExp() {

}

bioDerivatives* bioExprExp::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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

  bioUInt n = literalIds.size() ;
  bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  // if (childResult->f <= -10) {
  //   std::stringstream str ;
  //   str << "Low argument for exp " << childResult->f << "for " << child->print() ;
  //   throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  // }
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
