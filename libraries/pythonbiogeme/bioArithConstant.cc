//-*-c++-*------------------------------------------------------------
//
// File name : bioArithConstant.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed May  6 16:36:58 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patDisplay.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"

bioArithConstant::bioArithConstant(bioExpressionRepository* rep,
				   patULong par,
				   patReal aValue) : 
  bioArithElementaryExpression(rep, par),
  theValue(aValue) {

}
  
patString bioArithConstant::getOperatorName() const {
  stringstream str ;
  str << theValue ;
  return patString(str.str()) ; 
}

patReal bioArithConstant::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  //TODO maybe update lastValue and lastComputedLap
  return theValue ;
}

bioExpression* bioArithConstant::getDerivative(patULong aLiteralId, 
					       patError*& err) const {
  bioExpression* result = new bioArithConstant(theRepository,patBadId,0) ;
  return result ;
}

bioArithConstant* bioArithConstant::getDeepCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithConstant* theNode = new bioArithConstant(rep,patBadId,theValue) ;
  return theNode ;
}

bioArithConstant* bioArithConstant::getShallowCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithConstant* theNode = new bioArithConstant(rep,patBadId,theValue) ;
  return theNode ;
}



patBoolean bioArithConstant::isStructurallyZero() const {
  return (theValue == 0) ;
}

patBoolean bioArithConstant::isStructurallyOne() const {
  return (theValue == 1) ;
}

patBoolean bioArithConstant::isConstant() const {
  return patTRUE ;
}


patString bioArithConstant::getExpressionString() const {
  stringstream str ;
  str << theValue ;
  return patString(str.str()) ;

}

patString bioArithConstant::getExpression(patError*& err) const {
  stringstream str ;
  str << theValue ;
  return patString(str.str()) ;
}

patBoolean bioArithConstant::dependsOf(patULong aLiteralId) const {
  return patFALSE ;
}





bioFunctionAndDerivatives* bioArithConstant::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,
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
