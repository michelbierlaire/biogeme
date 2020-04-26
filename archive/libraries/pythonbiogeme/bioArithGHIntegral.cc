//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGHIntegral.cc
// Author :    Michel Bierlaire
// Date :      Tue Mar 30 17:33:15 2010
//
//--------------------------------------------------------------------

#include <sstream>

#include "patMath.h"
#include "bioParameters.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithGHIntegral.h"
#include "bioArithConstant.h"
#include "bioArithLiteral.h"
#include "bioArithMult.h"
#include "bioArithBinaryPlus.h"
#include "bioArithDivide.h"
#include "bioLiteralRepository.h"
#include "bioArithNotEqual.h"
#include "bioArithGaussHermite.h"
#include "bioGaussHermite.h"
#include "bioExpressionRepository.h"
#include "bioArithCompositeLiteral.h"

bioArithGHIntegral::bioArithGHIntegral(bioExpressionRepository* rep,
				   patULong par,
				   patULong left,
				   patString aLiteralName, 
				   patError*& err)
  : bioArithUnaryExpression(rep, par,left,err), literalName(aLiteralName) {
  pair<patULong,patULong> ids = 
    bioLiteralRepository::the()->getRandomVariable(literalName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  literalId = ids.second ;

}


bioArithGHIntegral::~bioArithGHIntegral() {}

patString bioArithGHIntegral::getOperatorName() const {
  return ("Integral") ;
}

patReal bioArithGHIntegral::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    bioArithGaussHermite theGh(child,vector<patULong>(),literalId,patFALSE, patFALSE) ;
    bioGaussHermite theGhAlgo(&theGh) ;
    vector<patReal> result = theGhAlgo.integrate(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    //return result[0] ;
    lastValue = result[0] ;
    lastComputedLap = currentLap;
  }
  return lastValue;
}


bioExpression* bioArithGHIntegral::getDerivative(patULong aLiteralId, 
					       patError*& err) const {

  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    WARNING(getInfo()) ;
    return NULL;
  }
  patString derivName = bioLiteralRepository::the()->getName(aLiteralId,err) ;

  if (aLiteralId == literalId) {
    stringstream str ;
    str << "Cannot compute the derivative with respect to " << derivName << " of an integral over " << literalName ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (!dependsOf(aLiteralId) || child->isStructurallyZero()) {
    bioExpression* result = theRepository->getZero() ;
    return result ;
  }
  
  bioExpression* leftDeriv = child->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }



  bioExpression* result = new bioArithGHIntegral(theRepository,
					       patBadId,
					       leftDeriv->getId(),
					       literalName,
					       err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}


bioArithGHIntegral* bioArithGHIntegral::getDeepCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  bioArithGHIntegral* theNode = 
    new bioArithGHIntegral(rep,
			 patBadId,
			 leftClone->getId(),
			 literalName,
			 err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}


bioArithGHIntegral* bioArithGHIntegral::getShallowCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithGHIntegral* theNode = 
    new bioArithGHIntegral(rep,
			 patBadId,
			 child->getId(),
			 literalName,
			 err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

patBoolean bioArithGHIntegral::isStructurallyZero() const {
  return (child->isStructurallyZero()) ;
}

patString bioArithGHIntegral::getExpressionString() const {
  stringstream str ;
  str << "Integ" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;

}



bioFunctionAndDerivatives* bioArithGHIntegral::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioArithGaussHermite theGh(child,literalIds,literalId,patTRUE,computeHessian) ;
  bioGaussHermite theGhAlgo(&theGh) ;
  vector<patReal> r = theGhAlgo.integrate(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  result.theFunction = r[0] ;
  for (patULong j = 0 ; j < literalIds.size() ; ++j) {
    if (patFinite(r[j+1])) {
      result.theGradient[j] = r[j+1] ;
    }
    else {
      result.theGradient[j] = patMaxReal ;
    }
  }
  if (result.theHessian != NULL && computeHessian) {
    patULong index = 1 + result.theGradient.size() ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	result.theHessian->setElement(i,j,r[index],err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	++index ;
      }
    }
  }
  return &result ;
}
