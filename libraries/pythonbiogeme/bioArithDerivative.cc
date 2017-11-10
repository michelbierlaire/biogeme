//-*-c++-*------------------------------------------------------------
//
// File name : bioArithDerivative.cc
// Author :    Michel Bierlaire
// Date :      Tue May 11 06:07:21 2010
//
//--------------------------------------------------------------------

#include <sstream>

#include "patMath.h"
#include "bioParameters.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithDerivative.h"
#include "bioArithConstant.h"
#include "bioArithLiteral.h"
#include "bioArithMult.h"
#include "bioArithBinaryPlus.h"
#include "bioArithDivide.h"
#include "bioLiteralRepository.h"
#include "bioArithNotEqual.h"
#include "bioExpressionRepository.h"

bioArithDerivative::bioArithDerivative(bioExpressionRepository* rep,
				       patULong par,
				       patULong left,
				       patString aLiteralName, 
				       patError*& err)
  : bioArithUnaryExpression(rep,par,left,err), literalName(aLiteralName)
{
  literalId = bioLiteralRepository::the()->getLiteralId(literalName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }

  bioExpression* theLeft = theRepository->getExpression(left) ;
  if (theLeft == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return ;
  }
  theDerivative = theLeft->getDerivative(literalId,err) ;
  relatedExpressions.push_back(theDerivative) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  theDerivativeId = theDerivative->getId() ;
  
  

//   DEBUG_MESSAGE("Derive the following expression wrt to " << literalName << " [" << literalId << "]") ;
//   DEBUG_MESSAGE("Expression: " << *leftChild) ;
//   DEBUG_MESSAGE("Result: " << *theDerivative) ;
}


bioArithDerivative::~bioArithDerivative() {}

patString bioArithDerivative::getOperatorName() const {
  return ("Derivative") ;
}

patReal bioArithDerivative::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
 
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){ 
    if (theDerivative == NULL) {
      err = new patErrNullPointer("bioExpression*") ;
      WARNING(err->describe()) ;
    }
    patReal result = theDerivative->getValue(prepareGradient, currentLap, err) ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    lastValue = result ;
    lastComputedLap = currentLap;
  }

  return lastValue;
}


bioExpression* bioArithDerivative::getDerivative(patULong aLiteralId, 
					       patError*& err) const {

  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  bioExpression* result = theDerivative->getDerivative(aLiteralId,err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}


bioArithDerivative* bioArithDerivative::getDeepCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  bioArithDerivative* theNode = 
    new bioArithDerivative(rep,patBadId,leftClone->getId(),literalName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}


bioArithDerivative* bioArithDerivative::getShallowCopy(bioExpressionRepository* rep,
						    patError*& err) const {
  bioArithDerivative* theNode = 
    new bioArithDerivative(rep,patBadId,child->getId(),literalName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

patBoolean bioArithDerivative::isStructurallyZero() const {
  return (child->isStructurallyZero()) ;
}
 
patString bioArithDerivative::getExpression(patError*& err) const {
  patString childResult = child->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  stringstream str ;
  str << "Derivative(" << childResult << ",'" << literalName << "')" ;
  return patString(str.str()) ;
}


patString bioArithDerivative::getExpressionString() const {
  stringstream str ;
  str << "Deriv" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithDerivative::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
							   patBoolean debugDeriv, patError*& err) {
  bioFunctionAndDerivatives* r = theDerivative->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return r ;
}

