//-*-c++-*------------------------------------------------------------
//
// File name : bioArithRandom.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Thu Jul 30 17:19:12 2009
//
//--------------------------------------------------------------------

#include "bioArithRandom.h"
#include "bioArithConstant.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "bioRandomDraws.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithRandom::bioArithRandom(bioExpressionRepository* rep,
			       patULong par, 
			       patULong individual,
			       patULong i, 
			       bioRandomDraws::bioDrawsType t,
			       patError*& err) :
  bioArithUnaryExpression(rep, par, individual,err),
  variableIdForDraws(i),
  type(t) {


}




bioArithRandom::~bioArithRandom() {
} ;

patString bioArithRandom::getOperatorName() const {
  stringstream str ;
  str << bioRandomDraws::the()->getTypeName(type) << "(" << variableIdForDraws << ")" ;
  return patString(str.str()) ;
}

patReal bioArithRandom::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (__draws == NULL) {
      stringstream str ;
      str << "Expression: " << *this << "\n" ;
      if (!isTop()) {
	bioExpression* parentPtr = this->getParent() ;
	if (parentPtr != NULL) { 
	  str << "Parent expression: " << *parentPtr << "\n" ;
	}
      }
      str << "No draws are available. The expression must be embedded inside a MonteCarlo statement: \ndrawresult = MonteCarlo(expression)\n" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal c = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    //  return (*__draws)[variableIdForDraws][patULong(c)] ;
    //return __draws[variableIdForDraws][patULong(c)] ;
    lastValue = __draws[variableIdForDraws][patULong(c)] ;
    lastComputedLap = currentLap;
  }
  return lastValue;

}

bioExpression* bioArithRandom::getDerivative(patULong aLiteralId, patError*& err) const {
  bioExpression* result = theRepository->getZero() ;
  return result ;
}

bioArithRandom* bioArithRandom::getDeepCopy(bioExpressionRepository* rep,
					    patError*& err) const {
  bioExpression* clone = child->getDeepCopy(rep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioArithRandom* theNewNode = new 
    bioArithRandom(rep,patBadId,clone->getId(),variableIdForDraws,type,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNewNode ;

}

bioArithRandom* bioArithRandom::getShallowCopy(bioExpressionRepository* rep,
					       patError*& err) const {
  bioArithRandom* theNewNode = new 
    bioArithRandom(rep,patBadId,child->getId(),variableIdForDraws,type,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNewNode ;

}


patULong bioArithRandom::getNumberOfOperations() const {
  return 0 ;
}

patString bioArithRandom::getExpressionString() const {
  stringstream str ;
  str << "&" ;
  str << bioRandomDraws::the()->getTypeName(type) << "(" << variableIdForDraws << ")" ;
  return patString(str.str()) ;

}

patString bioArithRandom::getExpression(patError*& err) const {
  stringstream str ;
  str << getOperatorName() ;
  return patString(str.str())  ;
}

patBoolean bioArithRandom::dependsOf(patULong aLiteralId) const {
  return patFALSE ;
}


bioFunctionAndDerivatives* bioArithRandom::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, patBoolean debugDeriv, patError*& err) {
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


vector<patULong> bioArithRandom::getListOfDraws(patError*& err) const {
  vector<patULong> result ;
  result.push_back(variableIdForDraws) ;
  return result ;
}

void bioArithRandom::checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) {
  if (!insideMonteCarlo) {
    err = new patErrMiscError("bioDraws must be used inside a MonteCarlo operator") ;
    WARNING(err->describe()) ;
    WARNING("The stack of involved expressions is reported below to help you locate where the problem is.")
    return ;
  }
}
