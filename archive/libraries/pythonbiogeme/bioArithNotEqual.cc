//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNotEqual.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:42:09 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithNotEqual.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithNotEqual::bioArithNotEqual(bioExpressionRepository* rep,
				   patULong par,
                                   patULong left, 
                                   patULong right,
				   patError*& err) 
  : bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithNotEqual::~bioArithNotEqual() {}

patString bioArithNotEqual::getOperatorName() const {
  return ("!=") ;
}


patReal bioArithNotEqual::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal l = leftChild->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal r = rightChild->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    //return ( l != r ) ;
    lastValue = ( l != r ) ;
    lastComputedLap = currentLap;
  }
  return lastValue;
}


bioExpression* bioArithNotEqual::getDerivative(patULong aLiteralId, 
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


bioArithNotEqual* bioArithNotEqual::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithNotEqual* theNode = 
    new bioArithNotEqual(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}


bioArithNotEqual* bioArithNotEqual::getShallowCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithNotEqual* theNode = 
    new bioArithNotEqual(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patString bioArithNotEqual::getExpressionString() const {
  stringstream str ;
  str << "!=" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

// bioArithListOfExpressions* bioArithNotEqual::getFunctionAndDerivatives(vector<patULong> literalIds, 
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


bioFunctionAndDerivatives* bioArithNotEqual::getNumericalFunctionAndGradient(vector<patULong> literalIds,
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
