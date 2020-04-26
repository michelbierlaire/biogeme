//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBinaryPlus.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 15:25:26 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithBinaryPlus.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"

bioArithBinaryPlus::bioArithBinaryPlus(bioExpressionRepository* rep,
				       patULong par,
                                       patULong left, 
                                       patULong right,
				       patError*& err) 
  : bioArithBinaryExpression(rep, par,left,right,err) {

}

bioArithBinaryPlus::~bioArithBinaryPlus() {}

patString bioArithBinaryPlus::getOperatorName() const {
  return ("+") ;
}

patReal bioArithBinaryPlus::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal left = leftChild->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal right = rightChild->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    lastValue = (left + right) ;
    lastComputedLap = currentLap;
  }
  return lastValue;

}

bioExpression* bioArithBinaryPlus::getDerivative(patULong aLiteralId,
						 patError*& err) const {
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  bioExpression* leftResult = leftChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioExpression* rightResult = rightChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (leftChild->dependsOf(aLiteralId)) {
    if (rightChild->dependsOf(aLiteralId)) {
      // Left variable, right variable
      bioExpression* result = new bioArithBinaryPlus(theRepository,patBadId,leftResult->getId(),rightResult->getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      return result ;
    }
    else {
      // Left variable, right constant
      return leftResult ;
    }
  }
  else {
    if (rightChild->dependsOf(aLiteralId)) {
      //Left constant, right variable
      return rightResult ;
    }
    else {
      //Left constant, right constant
      bioExpression* result = new bioArithConstant(theRepository,patBadId,0) ;
      return result ;
    }
  }


}

bioArithBinaryPlus* bioArithBinaryPlus::getDeepCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioExpression* leftClone(NULL) ;
  bioExpression* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (leftClone == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (rightClone == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  bioArithBinaryPlus* theNode = 
    new bioArithBinaryPlus(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
      
  return theNode ;
}

bioArithBinaryPlus* bioArithBinaryPlus::getShallowCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioArithBinaryPlus* theNode = 
    new bioArithBinaryPlus(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
      
  return theNode ;
}

patString bioArithBinaryPlus::getExpression(patError*& err) const {
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return patString();
  }
  stringstream str ;
  
  patString l = leftChild->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString r = rightChild->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }

  str << l << ' ' << getOperatorName() << ' ' << r  ;
  return patString(str.str()) ;
}


patString bioArithBinaryPlus::getExpressionString() const {
  stringstream str ;
  str << '+' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithBinaryPlus::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
							   patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }

  bioFunctionAndDerivatives* l = leftChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioFunctionAndDerivatives* r = 
    rightChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  result.theFunction =  l->theFunction + r->theFunction ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    result.theGradient[i] = l->theGradient[i] + r->theGradient[i] ;
  }

  // The second derivative matrix
  if (result.theHessian != NULL && computeHessian) {
    if (l->theHessian == NULL) {
      err = new patErrNullPointer("trHessian") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (r->theHessian == NULL) {
      err = new patErrNullPointer("trHessian") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal r1 = l->theHessian->getElement(i,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	patReal r2 = r->theHessian->getElement(i,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}

	patReal res = r1 + r2 ;
	if (performCheck) {
	  if (!patFinite(res)) {
	    res = patMaxReal ;
	  }
	}
	result.theHessian->setElement(i,j,res,err) ; 
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
  }

  return &result ;
}


