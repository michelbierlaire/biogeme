//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnaryMinus.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri May 15 09:41:41 2009
//
//--------------------------------------------------------------------

#include "bioArithUnaryMinus.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"

bioArithUnaryMinus::bioArithUnaryMinus(bioExpressionRepository* rep,
				       patULong par,
                                       patULong left,
				       patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err) {

}

bioArithUnaryMinus::~bioArithUnaryMinus() {
}

patString bioArithUnaryMinus::getOperatorName() const {
  return("-") ;
}

patReal bioArithUnaryMinus::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    if (child == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    
    patReal result = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    lastValue = -result ;
    lastComputedLap = currentLap;
  }
  return lastValue;
}

bioExpression* bioArithUnaryMinus::getDerivative(patULong aLiteralId, 
						 patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  bioExpression* leftResult = child->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* result = new bioArithUnaryMinus(theRepository,patBadId,leftResult->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
 return result ;
}

bioArithUnaryMinus* bioArithUnaryMinus::getDeepCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  bioArithUnaryMinus* theNode = 
    new bioArithUnaryMinus(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}


bioArithUnaryMinus* bioArithUnaryMinus::getShallowCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioArithUnaryMinus* theNode = 
    new bioArithUnaryMinus(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patString bioArithUnaryMinus::getExpressionString() const {
  stringstream str ;
  str << "$M" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}


bioFunctionAndDerivatives* bioArithUnaryMinus::getNumericalFunctionAndGradient(vector<patULong> literalIds,  patBoolean computeHessian, patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* c = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  result.theFunction =  - c->theFunction ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    result.theGradient[i] = - c->theGradient[i] ;
  }


  // The second derivative matrix
  if (result.theHessian != NULL && computeHessian) {
    if (c->theHessian == NULL) {
      err = new patErrNullPointer("trHessian") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal r = - c->theHessian->getElement(i,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	result.theHessian->setElement(i,j,r,err) ; 
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
  }

  return &result ;
}
