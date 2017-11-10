//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLessEqual.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:35:26 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithLessEqual.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithLessEqual::bioArithLessEqual(bioExpressionRepository* rep,
				     patULong par,
                                     patULong left, 
                                     patULong right,
				     patError*& err) 
  : bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithLessEqual::~bioArithLessEqual() {}

patString bioArithLessEqual::getOperatorName() const {
  return ("<=") ;
}


patReal bioArithLessEqual::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   {
  
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
    //return(left <= right) ;
    lastValue = (left <= right) ;
    lastComputedLap = currentLap;
  }

  return lastValue;
}


bioExpression* bioArithLessEqual::getDerivative(patULong aLiteralId, 
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


bioArithLessEqual* bioArithLessEqual::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithLessEqual* theNode = 
    new bioArithLessEqual(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}

bioArithLessEqual* bioArithLessEqual::getShallowCopy(bioExpressionRepository* rep,
						  patError*& err) const {
  bioArithLessEqual* theNode = 
    new bioArithLessEqual(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}



patString bioArithLessEqual::getExpressionString() const {
  stringstream str ;
  str << "<=" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

bioFunctionAndDerivatives* 
bioArithLessEqual::getNumericalFunctionAndGradient(vector<patULong> literalIds,
						   patBoolean computeHessian,
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
