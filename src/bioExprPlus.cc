//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPlus.cc
// @date   Fri Apr 13 10:27:14 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprPlus.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprPlus::bioExprPlus(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}

bioExprPlus::~bioExprPlus() {
}
  
bioDerivatives* bioExprPlus::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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
  bioDerivatives* leftResult = left->getValueAndDerivatives(literalIds,gradient,hessian) ;
  bioDerivatives* rightResult = right->getValueAndDerivatives(literalIds,gradient,hessian) ;
  if (rightResult == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"right result") ;
  }

  if (leftResult->f == 0.0) {
    theDerivatives->f = rightResult->f ;
  }
  else if (rightResult->f == 0.0) {
    theDerivatives->f = leftResult->f ;
  }
  else {
    theDerivatives->f = leftResult->f + rightResult->f ;
  }
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      if (leftResult->g[i] == 0.0) {
	theDerivatives->g[i] = rightResult->g[i] ;

      }
      else if (rightResult->g[i] == 0.0) {
	theDerivatives->g[i] = leftResult->g[i] ;
      }
      else {
	theDerivatives->g[i] = leftResult->g[i] + rightResult->g[i] ;
      }
      if (hessian) {
	for (bioUInt j = 0 ; j < n ; ++j) {
	  if (leftResult->h[i][j] == 0.0) {
	    theDerivatives->h[i][j] = rightResult->h[i][j] ;
	  }
	  else if (rightResult->h[i][j] == 0.0) {
	    theDerivatives->h[i][j] = leftResult->h[i][j] ;
	    
	  }
	  else {
	    theDerivatives->h[i][j] = leftResult->h[i][j] + rightResult->h[i][j] ;
	  }
	}
      }
    }
  }
  //  DEBUG_MESSAGE("Plus - Calculated: " << str.length() << " = " << theDerivatives->f) ;
  return theDerivatives ;
}

bioString bioExprPlus::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "+(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "+" << right->print(hp) << ")" ;
  }
  return str.str() ;
}
