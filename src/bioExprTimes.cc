//-*-c++-*------------------------------------------------------------
//
// File name : bioExprTimes.cc
// @date   Fri Apr 13 11:39:27 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprTimes.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprTimes::bioExprTimes(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}
bioExprTimes::~bioExprTimes() {

}

bioDerivatives* bioExprTimes::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						     bioBoolean gradient,
						     bioBoolean hessian) {


  //  DEBUG_MESSAGE("Derivatives of " << print());
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
  bioDerivatives* rightResult = NULL ;
  bioReal rightValue ;
  if (leftResult->f == 0.0 && !hessian) {
    // No need to calculate the derivatives of the other term
    rightValue = right->getValue() ;
  }
  else {
    rightResult = right->getValueAndDerivatives(literalIds,gradient,hessian) ;
    if (gradient) {
      if (rightResult->g.size() != literalIds.size()) {
	std::stringstream str ;
	str << "Wrong size for the result: " << rightResult->g.size() << " instead of " << literalIds.size() ;
	throw bioExceptions(__FILE__,__LINE__,str.str()) ;	
      }
    }
    rightValue = rightResult->f ;
  }

  if (leftResult->f == 0) {
    // l = 0
    if (rightValue == 0) {
      // l = 0, r = 0
      theDerivatives->f = 0.0 ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  theDerivatives->g[i] = 0.0 ;
	}
      }
    }
    else {
      // l = 0, r != 0
      theDerivatives->f = 0.0 ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (leftResult->g[i] == 0.0) {
	    theDerivatives->g[i] = 0.0 ;
	  }
	  else {
	    theDerivatives->g[i] = leftResult->g[i] * rightValue ;
	  }
	}
      }
    }
  }
  else {
    // l != 0 
    if (rightValue == 0) {
      // l != 0, r = 0
      theDerivatives->f = 0.0 ;
      if (rightResult == NULL) {
	throw bioExceptNullPointer(__FILE__,__LINE__,"Right result of multiplication") ;
      }
      
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (rightResult->g[i] == 0.0) {
	    theDerivatives->g[i] = 0.0 ;
	  }
	  else {
	    theDerivatives->g[i] = rightResult->g[i] * leftResult->f ;
	  }
	}
      }
    }
    else {
      // l != 0, r != 0
      theDerivatives->f = leftResult->f * rightValue ;
      if (gradient) {
	if (rightResult == NULL) {
	  throw bioExceptNullPointer(__FILE__,__LINE__,"Right result of multiplication") ;
	}
	for (bioUInt i = 0 ; i < n ; ++i) {
	  theDerivatives->g[i] = leftResult->g[i] * rightResult->f + 
	    rightResult->g[i] * leftResult->f ;
	}
      }
    }
  }

  if (hessian) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	bioReal v ;
	if (rightValue != 0.0) {
	  bioReal lhs = leftResult->h[i][j] ;
	  v = lhs * rightValue ;
	}
	else {
	  v = 0.0 ;
	}
	if (leftResult->f != 0) {
	  bioReal rhs = rightResult->h[i][j] ;
	  v += rhs * leftResult->f ;
	}
	if (rightResult == NULL) {
	  throw(bioExceptNullPointer(__FILE__,__LINE__,"rightResult")) ;
	}
	if (leftResult->g[i] != 0.0 && rightResult->g[j] != 0.0) {
	  v += leftResult->g[i] * rightResult->g[j] ;
	}
	if (leftResult->g[j] != 0.0 && rightResult->g[i] != 0.0) {
	  v += leftResult->g[j] * rightResult->g[i] ;
	}
	theDerivatives->h[i][j] = theDerivatives->h[j][i] = v ;
      }
    }    
  }

  
  return theDerivatives ;
}

bioString bioExprTimes::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "*(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "*" << right->print(hp) << ")" ;
  }
  return str.str() ;
}

