//-*-c++-*------------------------------------------------------------
//
// File name : bioArithAbs.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:56:09 2009
//
//--------------------------------------------------------------------

#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

#include "bioLiteralRepository.h"
#include "bioArithAbs.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h" 

bioArithAbs::bioArithAbs(bioExpressionRepository* rep, 
			 patULong par,
                         patULong c,
			 patError*& err) : bioArithUnaryExpression(rep,par,c,err)
{
}

bioArithAbs::~bioArithAbs() {}

patString bioArithAbs::getOperatorName() const {
  return ("abs") ;
}

patReal bioArithAbs::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {

  if(currentLap > lastComputedLap || currentLap == patLapForceCompute ){

    if (child == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal result = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    lastComputedLap = currentLap;
    lastValue = ((result>=0)?result:-result) ;
  
  }
  return lastValue ;
}


bioExpression* bioArithAbs::getDerivative(patULong aLiteralId, patError*& err) const {
  if (dependsOf(aLiteralId)) {
    err = new patErrMiscError("No derivative for absolute value") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  else {
    bioExpression* result = new bioArithConstant(theRepository,patBadId,0.0) ;
    return result ;
  }
}

bioArithAbs* bioArithAbs::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }

  bioArithAbs* theNode = 
    new bioArithAbs(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }
  return theNode ;
}


bioArithAbs* bioArithAbs::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioArithAbs* theNode = 
    new bioArithAbs(rep,patBadId,childId,err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }
  return theNode ;
}



patString bioArithAbs::getExpressionString() const {
  // stringstream str ;
  // str << "$A" ;
  // if (child != NULL) {
  //   str << '{' << child->getExpressionString() << '}' ;
  // }
  // return patString(str.str()) ;
  return patString() ;
}




bioFunctionAndDerivatives* bioArithAbs::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
							  patBoolean debugDeriv,  patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size(),0.0) ;
  }
  //TODO change forcecompute
  result.theFunction = getValue(patFALSE, patLapForceCompute, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return &result ;
}
