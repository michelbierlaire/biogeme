//-*-c++-*------------------------------------------------------------
//
// File name : bioArithAnd.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:36:07 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "bioLiteralRepository.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

#include "bioArithAnd.h"
#include "bioArithConstant.h"

bioArithAnd::bioArithAnd(bioExpressionRepository* rep,
			 patULong par,
                         patULong left, 
                         patULong right,
			 patError*& err) : 
  bioArithBinaryExpression(rep, par,left,right,err)
{
}

bioArithAnd::~bioArithAnd() {}

patString bioArithAnd::getOperatorName() const {
  return ("&&") ;
}

patReal bioArithAnd::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal left = leftChild->getValue(patFALSE, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal right = rightChild->getValue(patFALSE, currentLap,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    lastValue= left && right ;
    lastComputedLap = currentLap;
  }
  
  return lastValue;
  
}

bioExpression* bioArithAnd::getDerivative(patULong aLiteralId, 
					  patError*& err) const {
  if (dependsOf(aLiteralId)) {
    err = new patErrMiscError("No derivative for boolean functions") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  else {
    bioExpression* result = new bioArithConstant(theRepository,patBadId,0.0) ;
    return result ;
  }
}

bioArithAnd* bioArithAnd::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioExpression* leftClone(NULL) ;
  bioExpression* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(rep,err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(rep,err) ;
  }

  bioArithAnd* theNode = 
    new bioArithAnd(rep,patBadId,leftClone->getId(),rightClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }

  return theNode ;
}

bioArithAnd* bioArithAnd::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioArithAnd* theNode = 
    new bioArithAnd(rep,patBadId,leftChild->getId(),rightChild->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }

  return theNode ;
}



patString bioArithAnd::getExpressionString() const {
  stringstream str ;
  str << '&' ;
  if (leftChild != NULL) {
    str << '{' << leftChild->getExpressionString() << '}' ;
  }
  if (rightChild != NULL) {
    str << '{' << rightChild->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}


// bioArithListOfExpressions* bioArithAnd::getFunctionAndDerivatives(vector<patULong> literalIds, 
// 								  patError*& err) const {

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



bioFunctionAndDerivatives* bioArithAnd::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
							 patBoolean debugDeriv,   patError*& err) {
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
