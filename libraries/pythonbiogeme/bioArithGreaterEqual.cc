//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGreaterEqual.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:25:34 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithGreaterEqual.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithGreaterEqual::bioArithGreaterEqual(bioExpressionRepository* rep,
					   patULong par,
					   patULong left, 
					   patULong right,
					   patError*& err) 
  : bioArithBinaryExpression(rep,par,left,right,err)
{
}

bioArithGreaterEqual::~bioArithGreaterEqual() {}

patString bioArithGreaterEqual::getOperatorName() const {
  return (">=") ;
}


patReal bioArithGreaterEqual::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal left = leftChild->getValue(prepareGradient, currentLap, err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal right = rightChild->getValue(prepareGradient, currentLap, err)  ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    lastValue = (left >= right) ;
    lastComputedLap = currentLap;

  }

  return lastValue;
}


bioExpression* bioArithGreaterEqual::getDerivative(patULong aLiteralId, 
						   patError*& err) const {
  if (dependsOf(aLiteralId)) {
    err = new patErrMiscError("No derivative for boolean functions") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  else {
    bioExpression* result = theRepository->getZero() ;
    return result ;
  }

}


bioArithGreaterEqual* bioArithGreaterEqual::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithGreaterEqual* theNode = 
    new bioArithGreaterEqual(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  return theNode ;
}


bioArithGreaterEqual* bioArithGreaterEqual::getShallowCopy(bioExpressionRepository* rep,
							patError*& err) const {

  bioArithGreaterEqual* theNode = 
    new bioArithGreaterEqual(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  return theNode ;
}



patString bioArithGreaterEqual::getExpressionString() const {
  stringstream str ;
  str << ">=" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

bioFunctionAndDerivatives* bioArithGreaterEqual::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
									       patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size(),0.0) ;
  }
  result.theFunction = getValue(patFALSE, patLapForceCompute, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return &result ;

}
