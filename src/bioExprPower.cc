//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPower.cc
// @date   Fri Apr 13 12:20:46 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprPower.h"
#include <cmath>
#include <sstream>
#include "bioDebug.h"

bioExprPower::bioExprPower(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
  listOfChildren.push_back(l) ;
  listOfChildren.push_back(r) ;

}

bioExprPower::~bioExprPower() {

}

bioDerivatives* bioExprPower::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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
  
  if (rightResult->f == 0.0) {
    theDerivatives->f = 1.0 ;
  }
  else if (rightResult->f == 1.0) {
    theDerivatives->f = leftResult->f ;
  }
  else if (leftResult->f == 0) {
    theDerivatives->f = 0.0 ;
  }
  else {
    bioUInt rint = bioUInt(rightResult->f) ;
    if (bioReal(rint) == rightResult->f) {
      for (bioUInt i = 0 ; i < rint ; ++i) {
	if (i == 0) {
	  theDerivatives->f = leftResult->f ;
	}
	else {
	  theDerivatives->f *= leftResult->f ;
	}
      }
    }
    else {
      theDerivatives->f = pow(leftResult->f,rightResult->f) ;
    }
  }

  if (gradient) {
    std::vector<bioReal> G(n,0.0) ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] = 0.0 ;
      if (theDerivatives->f != 0.0) {
	if (leftResult->g[i] != 0.0 && rightResult->f != 0.0) {
	  bioReal term = leftResult->g[i] * rightResult->f / leftResult->f ;  
	  G[i] += term ;
	  
	}
	if (rightResult->g[i] != 0.0) {
	  G[i] += rightResult->g[i] * log(leftResult->f) ;
	}
	theDerivatives->g[i] = theDerivatives->f * G[i] ;
      }
    }
    if (hessian) {
      for (bioUInt i = 0 ; i < n ; ++i) {
	for (bioUInt j = i ; j < n ; ++j) {
	  bioReal v = G[i] * theDerivatives->g[j] ;
	  if (theDerivatives->f != 0 && rightResult != NULL) {
	    bioReal term(0.0) ;
	    bioReal hright = rightResult->h[i][j] ;
	    if (hright != 0.0) {
	      term += hright * log(leftResult->f) ;
	    }
	    if (leftResult->g[j] != 0.0 && rightResult->g[i] != 0.0) {
	      term += leftResult->g[j] * rightResult->g[i] / leftResult->f ;
	    } 
	    if (leftResult->g[i] != 0.0 && rightResult->g[j] != 0.0) {
	      term += leftResult->g[i] * rightResult->g[j] / leftResult->f ;
	    }
	    if (leftResult->g[i] != 0.0 && leftResult->g[j] != 0.0) {
	      bioReal asquare = leftResult->f * leftResult->f ;
	      term -= leftResult->g[i] * leftResult->g[j] * rightResult->f / asquare ;
	    }
	    bioReal hleft = leftResult->h[i][j] ;
	    if (hleft != 0.0) {
	      term += hleft * rightResult->f / leftResult->f ;
	    }
	    if (term != 0.0) {
	      v += term * theDerivatives->f ;
	    }
	  }
	  theDerivatives->h[i][j] = theDerivatives->h[j][i] = v ;
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprPower::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "^(" << left->print(hp) << "*" << right->print(hp) << ")" ;

  }
  else {
    str << "(" << left->print(hp) << "^" << right->print(hp) << ")" ;
  } 
  return str.str() ;
}
