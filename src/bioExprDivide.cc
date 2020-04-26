//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDivide.cc
// @date   Fri Apr 13 11:57:05 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprDivide.h"
#include <sstream>
#include "bioDebug.h"
#include "bioExceptions.h"

bioExprDivide::bioExprDivide(bioExpression* l, bioExpression* r) :
  left(l), right(r) {
    listOfChildren.push_back(l) ;
    listOfChildren.push_back(r) ;

}

bioExprDivide::~bioExprDivide() {

}

bioDerivatives* bioExprDivide::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						      bioBoolean gradient,
						      bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }

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
    rightValue = rightResult->f ;
  }

  bioReal rSquare = rightValue * rightValue ;
  bioReal rCube = rSquare * rightValue ;

  if (leftResult->f == 0.0) {
    // l = 0
    if (rightValue  == 0.0) {
      // l= 0, r = 0
      theDerivatives->f = 0.0 ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  theDerivatives->g[i] = 0.0 ;
	}
      }
    } 
    else if (rightValue == 1.0) {
      // l = 0, r = 1
      theDerivatives->f = 0.0 ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  theDerivatives->g[i] = leftResult->g[i] ;
	}
      }
    }
    else {
      // l=0, r != 0, r != 1
      theDerivatives->f =  0.0 ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (leftResult->g[i] == 0) {
	    theDerivatives->g[i] = 0.0 ;
	  }
	  else {
	    theDerivatives->g[i] = leftResult->g[i] / rightValue ;
	  }
	}
      }
    }
  }
  else {
    // l != 0
    if (rightValue == 0.0) {
      // l != 0, r = 0
      theDerivatives->f =  bioMaxReal ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  theDerivatives->g[i] = bioMaxReal ;
	}
      }
    }
    else if (rightValue == 1.0) {
      // l != 0, r = 1
      theDerivatives->f =  leftResult->f ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (leftResult->g[i] == 0.0) {
	    if (rightResult->g[i] == 0.0) {
	      theDerivatives->g[i] = 0.0 ;
	    } 
	    else {
	      theDerivatives->g[i] = - rightResult->g[i] * leftResult->f ;
	    }
	  }
	  else {
	    if (rightResult->g[i] == 0.0) {
	      theDerivatives->g[i] = leftResult->g[i] ;
	    } 
	    else {
	      theDerivatives->g[i] = leftResult->g[i] - rightResult->g[i] * leftResult->f ;
	    }
	  }
	}
      }
    }
    else {
      // l != 0, r != 0, r != 1
      theDerivatives->f =  leftResult->f / rightResult->f ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  bioReal num = (leftResult->g[i] * rightResult->f -
			 rightResult->g[i] * leftResult->f) ;
	  if (num != 0.0) {
	    theDerivatives->g[i] = num / rSquare ;
	  }
	  else {
	    theDerivatives->g[i] = 0.0 ;
	  }
	}
      }
    }
  }

  if (hessian) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	bioReal v ;
	if (leftResult->f != 0.0) {
	  bioReal rhs = rightResult->h[i][j] ;
	  v = - leftResult->f * rhs / rSquare ;
	  if (rightResult->g[i] != 0.0 && rightResult->g[j] != 0.0) {
	    v += 2.0 * leftResult->f * rightResult->g[i] * rightResult->g[j] / rCube ;
	  }
	}
	else {
	  v = 0.0 ;
	}
	bioReal lhs = leftResult->h[i][j] ;
	if (lhs != 0.0) {
	  v += lhs / rightValue ;
	}
	if (rightResult != NULL) {
	  if (leftResult->g[i] != 0.0 && rightResult->g[j] != 0.0) {
	    v -=  leftResult->g[i] * rightResult->g[j] / rSquare ;
	  }
	  if (leftResult->g[j] != 0.0 && rightResult->g[i] != 0.0) {
	    v -=  leftResult->g[j] * rightResult->g[i] / rSquare ;
	  }
	}
	theDerivatives->h[i][j] = theDerivatives->h[j][i] = v ;
      }
    }
  }
  

  return theDerivatives ;
}

bioString bioExprDivide::print(bioBoolean hp) const {
  std::stringstream str ;
  if (hp) {
    str << "/(" << left->print(hp) << "," << right->print(hp) << ")" ;
  }
  else {
    str << "(" << left->print(hp) << "/" << right->print(hp) << ")" ;
  }
  return str.str() ;
}

