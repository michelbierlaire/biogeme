//-*-c++-*------------------------------------------------------------
//
// File name : bioArithEqual.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:10:28 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithEqual.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithEqual::bioArithEqual(bioExpressionRepository* rep,
			     patULong par,
                             patULong left, 
                             patULong right,
			     patError*& err) : 
  bioArithBinaryExpression(rep,par,left,right,err)
{
}

bioArithEqual::~bioArithEqual() {}

patString bioArithEqual::getOperatorName() const {
  return ("==") ;
}


patReal bioArithEqual::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {

  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }

    patReal l = leftChild->getValue(patFALSE, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal r = rightChild->getValue(patFALSE, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    //return ( l == r ) ;
    lastValue = (l==r);
    lastComputedLap = currentLap;
  }

  return lastValue;
}


bioExpression* bioArithEqual::getDerivative(patULong aLiteralId, 
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


bioArithEqual* bioArithEqual::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithEqual* theNode = 
    new bioArithEqual(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithEqual* bioArithEqual::getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const {
  bioArithEqual* theNode = 
    new bioArithEqual(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}




patString bioArithEqual::getExpressionString() const {
  stringstream str ;
  str << '=' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}


// bioArithListOfExpressions* bioArithEqual::getFunctionAndDerivatives(vector<patULong> literalIds, 
// 								   patError*& err) const {
  
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



bioFunctionAndDerivatives* bioArithEqual::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
