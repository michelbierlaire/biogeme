//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNormalCdf.cc
// Author :    Michel Bierlaire
// Date :      Tue Jun  8 18:00:07 2010
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithNormalCdf.h"
#include "bioArithDivide.h"
#include "bioArithMult.h"
#include "bioArithConstant.h"
#include "bioArithExp.h"
#include "bioArithUnaryMinus.h"

bioArithNormalCdf::bioArithNormalCdf(bioExpressionRepository* rep,
				     patULong par,
				     patULong left,
				     patError*& err) 
  : bioArithUnaryExpression(rep,par,left,err),invSqrtTwoPi(0.3989422804)
{
    pi = 4.0 * atan(1.0) ;
}

bioArithNormalCdf::~bioArithNormalCdf() {}

patString bioArithNormalCdf::getOperatorName() const {
  return ("normalCdf") ;
}

patReal bioArithNormalCdf::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (child == NULL) {
      err = new patErrNullPointer("patArithNode") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal childresult = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal result = patNormalCdf::the()->compute(childresult,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    if (result > 1.0) {
      result = 1.0 ;
    }
    if (result < 0.0) {
      result = 0.0 ;
    }
    lastValue = result ;
    lastComputedLap = currentLap ;
    return lastValue;
  }else{
    return lastValue;
  }
}


bioExpression* bioArithNormalCdf::getDerivative(patULong aLiteralId, patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  

  bioExpression* value1 = child->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* value2 = child->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* deriv = child->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // value * value 

  bioExpression* y2 = new bioArithMult(theRepository,patBadId,value1->getId(),value2->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // (value * value) / 2

  bioExpression* two = new bioArithConstant(theRepository,patBadId,2.0) ;

  bioExpression* d = new bioArithDivide(theRepository,patBadId,y2->getId(),two->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // - (value * value) / 2

  bioExpression* minusd = new bioArithUnaryMinus(theRepository,patBadId,d->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // exp(- (value * value) / 2)

  bioExpression* expo = new bioArithExp(theRepository,patBadId,minusd->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  // oneDivSqrtTwoPi * exp(- (value * value) / 2)


  bioExpression*  oneDivSqrtTwoPi =  new bioArithConstant(theRepository,patBadId,1.0/sqrt(2*pi)) ;

  bioExpression* m = new bioArithMult(theRepository,patBadId,oneDivSqrtTwoPi->getId(),expo->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* result = new bioArithMult(theRepository,patBadId,m->getId(),deriv->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  return result ;
}


bioArithNormalCdf* bioArithNormalCdf::getDeepCopy(bioExpressionRepository* rep,
						  patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }

  bioArithNormalCdf* theNode = 
    new bioArithNormalCdf(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}

bioArithNormalCdf* bioArithNormalCdf::getShallowCopy(bioExpressionRepository* rep,
						  patError*& err) const {
  bioArithNormalCdf* theNode = 
    new bioArithNormalCdf(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}



patString bioArithNormalCdf::getExpressionString() const {
  stringstream str ;
  str << "$NCDF" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}



bioFunctionAndDerivatives* bioArithNormalCdf::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {


  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* c = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  result.theFunction = patNormalCdf::the()->compute(c->theFunction,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  patReal thePdf = invSqrtTwoPi * exp(- c->theFunction * c->theFunction / 2.0) ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    if (c->theGradient[i] == 0.0) {
      result.theGradient[i] = 0.0 ;
    }
    else {
      result.theGradient[i] = thePdf * c->theGradient[i] ;
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
	if (h != 0.0) {
	  v += thePdf * h ;
	}
	if (c->theFunction != 0.0 && 
	    c->theGradient[i] != 0 && 
	    c->theGradient[j] != 0) {
	  v -= thePdf * c->theFunction * c->theGradient[i] * c->theGradient[j] ;
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
