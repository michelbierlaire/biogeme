//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMult.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 20:44:04 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

#include "bioArithMult.h"
#include "bioArithBinaryPlus.h"
#include "bioArithConstant.h"
#include "patError.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"
#include "bioArithCompositeLiteral.h"

bioArithMult::bioArithMult(bioExpressionRepository* rep,
			   patULong par,
			   patULong left, 
			   patULong right,
			   patError*& err) : 
  bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithMult::~bioArithMult() {}

patString bioArithMult::getOperatorName() const {
  return ("*") ;
}

patReal bioArithMult::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {

  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    patReal leftVal = leftChild->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    if (leftVal == 0.0) {
      lastValue = 0.0;
      //return 0.0 ;
    }else{

      patReal rightVal = rightChild->getValue(prepareGradient, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal();
      }
      
      if (rightVal == 0.0) {
        lastValue = 0.0;
        //return 0.0 ;
      }else{
        lastValue = leftVal * rightVal ;
      }

    }

    lastComputedLap = currentLap;
  }
  return lastValue ;
}

bioExpression* bioArithMult::getDerivative(patULong aLiteralId, 
					   patError*& err) const {
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }

  
  bioExpression* leftValue = leftChild->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* rightValue = rightChild->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  patBoolean leftOne(patFALSE) ;
  bioExpression* leftDeriv = leftChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  if (leftDeriv->isConstant()) {
    patReal v = leftDeriv->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    if (v == 1) {
      leftOne = patTRUE ;
    }
  }
  patBoolean rightOne(patFALSE) ;
  bioExpression* rightDeriv = rightChild->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  if (rightDeriv->isConstant()) {
    patReal v = rightDeriv->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    if (v == 1) {
      rightOne = patTRUE ;
    }
  }

  

  bioExpression* l ;
  if (leftOne) {
    l = rightValue->getDeepCopy(theRepository,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
  }
  else {
    l = new bioArithMult(theRepository,patBadId,leftDeriv->getId(),rightValue->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
  }
  bioExpression* r ;
  if (rightOne) {
    r = leftValue->getDeepCopy(theRepository,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
  }
  else {
    r = new bioArithMult(theRepository,patBadId,leftValue->getId(),rightDeriv->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
  }
  if (leftChild->dependsOf(aLiteralId)) {
    if (rightChild->dependsOf(aLiteralId)) {
      // Left variable, right variable
      bioExpression* result = new bioArithBinaryPlus(theRepository,patBadId,l->getId(),r->getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL;
      }
      return result ;
    }
    else {
      // Left variable, right constant
      return l ;
    }
  }
  else {
    if (rightChild->dependsOf(aLiteralId)) {
      //Left constant, right variable
      return r ;
    }
    else {
      //Left constant, right constant
      bioExpression* result = theRepository->getZero() ;
      return result ;
    }
  }


}

bioArithMult* bioArithMult::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithMult* theNode = 
    new bioArithMult(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  

  return theNode ;
}

bioArithMult* bioArithMult::getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioArithMult* theNode = 
    new bioArithMult(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  

  return theNode ;
}

patString bioArithMult::getExpressionString() const {
  stringstream str ;
  str << '*' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

patBoolean bioArithMult::isStructurallyZero() const {
  if (leftChild->isStructurallyZero()) {
    return patTRUE ;
  }
  if (rightChild->isStructurallyZero()) {
    return patTRUE ;
  }
  return patFALSE ;
}


bioFunctionAndDerivatives* bioArithMult::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
									patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* l = leftChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioFunctionAndDerivatives* r = rightChild->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (l->theFunction == 0) {
    // l = 0
    patReal r = rightChild->getValue(patFALSE, patLapForceCompute, err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (r == 0) {
      // l = 0, r = 0
      result.theFunction =  0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = 0.0 ;
      }
    }
    else {
      // l = 0, r != 0
      result.theFunction =  0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (l->theGradient[i] == 0.0) {
	  result.theGradient[i] = 0.0 ;
	}
	else {
	  result.theGradient[i] = l->theGradient[i] * r ;
	}
      }
    }
  }
  else {
    // l != 0 
    if (r->theFunction == 0) {
      // l != 0, r = 0
      result.theFunction = 0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (r->theGradient[i] == 0.0) {
	  result.theGradient[i] = 0.0 ;
	}
	else {
	  result.theGradient[i] = r->theGradient[i] * l->theFunction ;
	}
      }
    }
    else {
      // l != 0, r = 0
      result.theFunction =  l->theFunction * r->theFunction ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = l->theGradient[i] * r->theFunction + 
	  r->theGradient[i] * l->theFunction ;
      }
      
    }
  }

  if (performCheck) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (!patFinite(result.theGradient[i])) {
	result.theGradient[i] = patMaxReal ;
      }
    }
  }

  if (result.theHessian != NULL && computeHessian) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal v ;
	if (r->theFunction != 0.0) {
	  patReal lhs = l->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  v = lhs * r->theFunction ;
	}
	else {
	  v = 0.0 ;
	}
	if (l->theFunction != 0) {
	  patReal rhs = r->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  v += rhs * l->theFunction ;

	}
	if (l->theGradient[i] != 0.0 && r->theGradient[j] != 0.0) {
	  v += l->theGradient[i] * r->theGradient[j] ;
	}
	if (l->theGradient[j] != 0.0 && r->theGradient[i] != 0.0) {
	  v += l->theGradient[j] * r->theGradient[i] ;
	}
	if (performCheck) {
	  if (!patFinite(v)) {
	    v = patMaxReal ;
	  }
	}
	result.theHessian->setElement(i,j,v,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }    
  }
  return &result ;
}

