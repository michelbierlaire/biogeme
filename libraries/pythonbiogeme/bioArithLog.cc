//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLog.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 14:05:01 2009
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <sstream>
#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithLog.h"
#include "bioArithConstant.h"
#include "bioArithDivide.h"
#include "bioArithCompositeLiteral.h"
#ifdef DEBUG
#include "bioParameters.h"
#endif
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithLog::bioArithLog(bioExpressionRepository* rep,
			 patULong par,
                         patULong left,
			 patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err) {

}

bioArithLog::~bioArithLog() {}

patString bioArithLog::getOperatorName() const {
  return ("log") ;
}

patReal bioArithLog::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (child == NULL) {
      err = new patErrNullPointer("patArithNode") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    patReal childValue = child->getValue(prepareGradient, currentLap, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    if (childValue == 1.0) {
      lastValue = 0.0;
      //return 0.0 ;
    } else {
      if (childValue == 0.0) {
	lastValue = -patMaxReal/2.0 ;
      }
      else {
	lastValue = log(childValue) ;
      }
    }

    lastComputedLap = currentLap;
  }
  return lastValue;
}


bioExpression* bioArithLog::getDerivative(patULong aLiteralId, patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  bioExpression* value = child->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* deriv = child->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* result = new bioArithDivide(theRepository,patBadId,deriv->getId(),value->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}


bioArithLog* bioArithLog::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  bioArithLog* theNode = 
    new bioArithLog(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}


bioArithLog* bioArithLog::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioArithLog* theNode = 
    new bioArithLog(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patString bioArithLog::getExpressionString() const {
  stringstream str ;
  str << "$L" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}


bioFunctionAndDerivatives* bioArithLog::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* c = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (c->theFunction == 0) {
    result.theFunction = -patMaxReal/2.0 ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (c->theGradient[i] == 0) {
	result.theGradient[i] = 0.0 ;
      }
      else {
	if (c->theGradient[i] != 0.0) {
	  result.theGradient[i] = c->theGradient[i] / c->theFunction ;
	  if (performCheck) {
	    if (!patFinite(result.theGradient[i])) {
	      DEBUG_MESSAGE(c->theGradient[i] << " / " << c->theFunction << " = " << result.theGradient[i]) ;
	      
	    }
	  }
	}
	else {
	  result.theGradient[i] = 0.0 ;
	}
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
	  if (h == 0.0 && (c->theGradient[i] == 0.0 || c->theGradient[j] == 0.0)) {
	    result.theHessian->setElement(i,j,0,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	  }
	  else {
	    result.theHessian->setElement(i,j,patMaxReal,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }

	  }
	}
      }
    }
  }
  else {
    if (c->theFunction == 1) {
      result.theFunction =  0.0 ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = c->theGradient[i] ;
      }
    }
    else {
      result.theFunction =  log(c->theFunction) ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (c->theGradient[i] != 0.0) {
	  result.theGradient[i] = c->theGradient[i] / c->theFunction ;
	}
	else {
	  result.theGradient[i] = 0.0 ;
	}
      }
    }
    if (result.theHessian != NULL && computeHessian) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  patReal v = -result.theGradient[i] * c->theGradient[j] / c->theFunction ;
	  //	DEBUG_MESSAGE("v["<<i<<"]["<<j<<"] = " << v) ;
	  patReal hs = c->theHessian->getElement(i,j,err) ;
	  //	DEBUG_MESSAGE("hs["<<i<<"]["<<j<<"] = " << v) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  if (hs != 0.0) {
	    v += hs / c->theFunction ;
	  }
	  //	DEBUG_MESSAGE("v["<<i<<"]["<<j<<"] = " << v) ;
	  if (!patFinite(v)) {
	    DEBUG_MESSAGE("Expr: " << getExpressionString()) ;
	    DEBUG_MESSAGE("c->theFunction: " << c->theFunction) ;
	    DEBUG_MESSAGE("H("<<i<<","<<j<<")="<<v) ;
	  }
	  result.theHessian->setElement(i,j,v,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}
      }
    }
  }
 
#ifdef  DEBUG
    if (debugDeriv != 0) {
      bioFunctionAndDerivatives* findiff = 
	getNumericalFunctionAndFinDiffGradient(literalIds, err)  ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      patReal compare = result.compare(*findiff,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      patReal tolerance = bioParameters::the()->getValueReal("toleranceCheckDerivatives",err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      if (compare >= tolerance) {
	DEBUG_MESSAGE("Analytical: " << result) ;
	DEBUG_MESSAGE("FinDiff   : " << *findiff) ;
	WARNING("Error " << compare << " in " << *this);
	err = new patErrMiscError("Error with derivatives") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
#endif
  return &result ;  
}

