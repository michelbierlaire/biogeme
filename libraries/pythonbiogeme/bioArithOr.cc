//-*-c++-*------------------------------------------------------------
//
// File name : bioArithOr.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:36:07 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithOr.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithOr::bioArithOr(bioExpressionRepository* rep,
		       patULong par,
                       patULong left, 
                       patULong right,
		       patError*& err) : 
  bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithOr::~bioArithOr() {}

patString bioArithOr::getOperatorName() const {
  return ("or") ;
}

patReal bioArithOr::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  
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
    
    lastValue = left || right ;
    lastComputedLap = currentLap ;
    return lastValue;
  }else{
    return lastValue;
  }
}

bioExpression* bioArithOr::getDerivative(patULong aLiteralId, patError*& err) const {
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

bioArithOr* bioArithOr::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithOr* theNode = 
    new bioArithOr(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithOr* bioArithOr::getShallowCopy(bioExpressionRepository* rep,
				    patError*& err) const {
  bioArithOr* theNode = 
    new bioArithOr(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patString bioArithOr::getExpressionString() const {
  stringstream str ;
  str << '|' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

// bioArithListOfExpressions* bioArithOr::getFunctionAndDerivatives(vector<patULong> literalIds, 
// 								 patError*& err) const {
  
//   vector<patULong> theExpressions ;
//   theExpressions.push_back(getId()) ;
//   bioArithConstant* zero = new bioArithConstant(patBadId,0.0) ;
//   for (patULong i = 1 ; i <= literalIds.size() ; ++i) {
//     if (dependsOf(literalIds[i])) {
//       err = new patErrMiscError("No derivative for absolute value") ;
//       WARNING(err->describe()) ;
//       return NULL ;
//     }
//     theExpressions.push_back(zero->getId()) ;
//   }
  
//   bioArithListOfExpressions* result = new bioArithListOfExpressions(NULL, theExpressions) ;
//   return result ;
// }


bioFunctionAndDerivatives* bioArithOr::getNumericalFunctionAndGradient(vector<patULong> literalIds,
patBoolean computeHessian,
							  patBoolean debugDeriv,  patError*& err) {
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
