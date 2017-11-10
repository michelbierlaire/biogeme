//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGreater.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:25:34 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithGreater.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"

#include "bioLiteralRepository.h"
bioArithGreater::bioArithGreater(bioExpressionRepository* rep,
				 patULong par,
                                 patULong left, 
                                 patULong right,
				 patError*& err) 
  : bioArithBinaryExpression(rep,par,left,right,err)
{
}

bioArithGreater::~bioArithGreater() {}


patString bioArithGreater::getOperatorName() const {
  return (">") ;
}


patReal bioArithGreater::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  
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
    //return (left > right)?1.0:0.0 ;
    lastValue = (left > right)?1.0:0.0 ;
    lastComputedLap = currentLap;
  }
  return lastValue;

}


bioExpression* bioArithGreater::getDerivative(patULong aLiteralId, 
					      patError*& err) const {
  if (dependsOf(aLiteralId)) {
    stringstream str ;
    patString theParam = bioLiteralRepository::the()->getName(aLiteralId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    str << "Differentiation of a boolean function with respect to " << theParam << ": " << *this ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  else {
    bioExpression* result = theRepository->getZero() ;
    return result ;
  }
}


bioArithGreater* bioArithGreater::getDeepCopy(bioExpressionRepository* rep,
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

  bioArithGreater* theNode = 
    new bioArithGreater(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}

bioArithGreater* bioArithGreater::getShallowCopy(bioExpressionRepository* rep,
					      patError*& err) const {
  bioArithGreater* theNode = 
    new bioArithGreater(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}






patString bioArithGreater::getExpressionString() const {
  stringstream str ;
  str << ">o" ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}

// bioArithListOfExpressions* bioArithGreater::getFunctionAndDerivatives(vector<patULong> literalIds, 
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

bioFunctionAndDerivatives* bioArithGreater::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
