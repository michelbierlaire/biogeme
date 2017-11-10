//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnifRandom.cc
// Author :    Michel Bierlaire
// Date :      Wed May 13 16:13:27 2015
//
//--------------------------------------------------------------------

#include "bioArithUnifRandom.h"
#include "bioArithConstant.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "bioRandomDraws.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithUnifRandom::bioArithUnifRandom(bioExpressionRepository* rep,
				       patULong par, 
				       patULong individual,
				       patULong i, 
				       bioRandomDraws::bioDrawsType t,
				       patError*& err) :
  bioArithUnaryExpression(rep, par, individual,err),
  variableIdForDraws(i),
  type(t) {
  
  
}




bioArithUnifRandom::~bioArithUnifRandom() {
} ;

patString bioArithUnifRandom::getOperatorName() const {
  stringstream str ;
  str << "Uniform draws used for " << bioRandomDraws::the()->getTypeName(type) << "(" << variableIdForDraws << ")" ;
  return patString(str.str()) ;
}

patReal bioArithUnifRandom::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (__unifdraws == NULL) {
      stringstream str ;
      str << "No draws are available. The expression must be embedded inside a draw iterator using the following syntax: \ndrawIterator('drawIter')\nresult = Sum(expression,'drawIter')\n" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal c = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
    lastValue = __unifdraws[variableIdForDraws][patULong(c)] ;
    lastComputedLap = currentLap;
  }
  return lastValue;

}

bioExpression* bioArithUnifRandom::getDerivative(patULong aLiteralId, patError*& err) const {
  bioExpression* result = theRepository->getZero() ;
  return result ;
}

bioArithUnifRandom* bioArithUnifRandom::getDeepCopy(bioExpressionRepository* rep,
					    patError*& err) const {
  bioExpression* clone = child->getDeepCopy(rep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioArithUnifRandom* theNewNode = new 
    bioArithUnifRandom(rep,patBadId,clone->getId(),variableIdForDraws,type,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNewNode ;

}

bioArithUnifRandom* bioArithUnifRandom::getShallowCopy(bioExpressionRepository* rep,
					       patError*& err) const {
  bioArithUnifRandom* theNewNode = new 
    bioArithUnifRandom(rep,patBadId,child->getId(),variableIdForDraws,type,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNewNode ;

}


patULong bioArithUnifRandom::getNumberOfOperations() const {
  return 0 ;
}

patString bioArithUnifRandom::getExpressionString() const {
  stringstream str ;
  str << "&U" ;
  str << bioRandomDraws::the()->getTypeName(type) << "(" << variableIdForDraws << ")" ;
  return patString(str.str()) ;

}

patString bioArithUnifRandom::getExpression(patError*& err) const {
  return getOperatorName() ;
}

patBoolean bioArithUnifRandom::dependsOf(patULong aLiteralId) const {
  return patFALSE ;
}


bioFunctionAndDerivatives* bioArithUnifRandom::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, patBoolean debugDeriv, patError*& err) {
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
