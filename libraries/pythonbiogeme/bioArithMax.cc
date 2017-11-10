//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMax.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 16:47:49 2010
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patMath.h"
#include "bioArithMax.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithMax::bioArithMax(bioExpressionRepository* rep, 
			 patULong par,
			 patULong left, 
			 patULong right,
			 patError*& err) : bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithMax::~bioArithMax() {}

patString bioArithMax::getOperatorName() const {
  return ("max") ;
}


patReal bioArithMax::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
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
    lastValue = patMax(left,right) ;
    lastComputedLap = currentLap;
    return lastValue ;
  }else{
    return lastValue ;
  }
}


bioExpression* bioArithMax::getDerivative(patULong aLiteralId, 
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


bioArithMax* bioArithMax::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithMax* theNode = 
    new bioArithMax(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithMax* bioArithMax::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioArithMax* theNode = 
    new bioArithMax(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}




patString bioArithMax::getExpressionString() const {
  stringstream str ;
  str << "max" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithMax::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
