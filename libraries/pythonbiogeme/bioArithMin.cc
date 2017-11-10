//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMin.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 16:51:15 2010
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patMath.h"
#include "patErrMiscError.h"
#include "bioArithMin.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithMin::bioArithMin(bioExpressionRepository* rep,
			 patULong par,
			 patULong left, 
			 patULong right,
			 patError*& err) 
  : bioArithBinaryExpression(rep,par,left,right,err)
{

  //  DEBUG_MESSAGE("CREATE MIN OPERATOR: min(" << *left << "," << *right << ")") ;
}

bioArithMin::~bioArithMin() {}

patString bioArithMin::getOperatorName() const {
  return ("min") ;
}


patReal bioArithMin::getValue(patBoolean prepareGradient, patULong currentLap,
			      patError*& err) {
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

    lastValue = patMin(left,right) ;
    lastComputedLap = currentLap;
    return lastValue ;
  }else{
    return lastValue;
  }
}


bioExpression* bioArithMin::getDerivative(patULong aLiteralId, 
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


bioArithMin* bioArithMin::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithMin* theNode = 
    new bioArithMin(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}


bioArithMin* bioArithMin::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioArithMin* theNode = 
    new bioArithMin(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}




patString bioArithMin::getExpressionString() const {
  stringstream str ;
  str << "min" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithMin::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
