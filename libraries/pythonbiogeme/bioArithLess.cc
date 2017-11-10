//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLess.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:28:36 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithLess.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithLess::bioArithLess(bioExpressionRepository* rep,
			   patULong par,
                           patULong left, 
                           patULong right,
			   patError*& err) 
  : bioArithBinaryExpression(rep,par,left,right,err)
{
}

bioArithLess::~bioArithLess() {}

patString bioArithLess::getOperatorName() const {
  return ("<") ;
}


patReal bioArithLess::getValue(patBoolean prepareGradient, patULong currentLap, 
			       patError*& err)  {
  if( currentLap > lastComputedLap || currentLap == patLapForceCompute){ 
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

    lastValue = (left < right);
    lastComputedLap = currentLap;
    return lastValue ;
  }else{
    return lastValue;
  }
}


bioExpression* bioArithLess::getDerivative(patULong aLiteralId, 
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


bioArithLess* bioArithLess::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithLess* theNode = 
    new bioArithLess(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithLess* bioArithLess::getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioArithLess* theNode = 
    new bioArithLess(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}




patString bioArithLess::getExpressionString() const {
  stringstream str ;
  str << '<' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithLess::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
