//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNormalPdf.cc
// Author :    Michel Bierlaire
// Date :      Mon Jun 27 18:22:36 2011
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "patMath.h"
#include "patDisplay.h"
#include "bioArithNormalPdf.h"
#include "bioArithUnaryMinus.h"
#include "bioExpressionRepository.h"
#include "bioArithMult.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

bioArithNormalPdf::bioArithNormalPdf(bioExpressionRepository* rep,
				     patULong par,
				     patULong left,
				     patError*& err) :
  bioArithUnaryExpression(rep,par,left,err), invSqrtTwoPi(0.3989422804) {

}

bioArithNormalPdf::~bioArithNormalPdf() {

}
patString bioArithNormalPdf::getOperatorName() const {
  return patString("normalPdf") ;
}

patReal bioArithNormalPdf::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (child == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal result = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    //return(invSqrtTwoPi * exp(-result * result / 2.0)) ;
    lastValue = (invSqrtTwoPi * exp(-result * result / 2.0)) ;
    lastComputedLap = currentLap;
  }
  return lastValue;
}

bioExpression* bioArithNormalPdf::getDerivative(patULong aLiteralId, patError*& err) const {
  if (child->dependsOf(aLiteralId)) {
    bioExpression* m1 = this->getDeepCopy(theRepository,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    bioExpression* m2 = child->getDeepCopy(theRepository,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    bioExpression* m3 = child->getDerivative(aLiteralId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    bioExpression* m1m2 = new bioArithMult(theRepository,
					   patBadId,
					   m1->getId(),
					   m2->getId(),
					   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    bioExpression* m1m2m3 = new bioArithMult(theRepository,
					     patBadId,
					     m1m2->getId(),
					     m3->getId(),
					     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    bioExpression* result = new bioArithUnaryMinus(theRepository,
						   patBadId,
						   m1m2m3->getId(),
						   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    return result ;
						   
  }
  else {
    bioExpression* theExpr = theRepository->getZero() ;
    return theExpr ;
  }

}

bioArithNormalPdf* bioArithNormalPdf::getDeepCopy(bioExpressionRepository* rep,
						  patError*& err) const {

  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  bioArithNormalPdf* theNode = 
    new bioArithNormalPdf(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithNormalPdf* bioArithNormalPdf::getShallowCopy(bioExpressionRepository* rep,
						     patError*& err) const {

  bioArithNormalPdf* theNode = 
    new bioArithNormalPdf(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

patString bioArithNormalPdf::getExpressionString() const {
  stringstream str ;
  str << "$NPDF" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}

bioFunctionAndDerivatives* bioArithNormalPdf::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {

  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* c = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  result.theFunction =  invSqrtTwoPi * exp(- c->theFunction * c->theFunction / 2.0) ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    if (c->theGradient[i] == 0.0) {
      result.theGradient[i] = 0.0 ;
    }
    else {
      result.theGradient[i] = - result.theFunction * c->theFunction * c->theGradient[i] ;
    }
  }
  if (result.theHessian != NULL && computeHessian) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal h = c->theHessian->getElement(i,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	patReal v(0.0) ;
	if (result.theFunction != 0.0 && 
	    c->theFunction != 0.0 &&  
	    h != 0.0) {
	  v -=  result.theFunction * c->theFunction * h ;
	}
	if (result.theGradient[j] != 0.0 && 
	    c->theFunction != 0.0 && 
	    c->theGradient[i] != 0.0) {
	  v -= result.theGradient[j] * c->theFunction * c->theGradient[i] ;
	}
	if (result.theFunction != 0.0 && 
	    c->theGradient[i] != 0.0 &&
	    c->theGradient[j] != 0.0) {
	  v -= result.theFunction * c->theGradient[i] * c->theGradient[j] ;
	}
	result.theHessian->setElement(i,j,v,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
  }
  return &result ;

}
