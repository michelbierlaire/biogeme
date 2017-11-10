//-*-c++-*------------------------------------------------------------
//
// File name : bioArithExp.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 14:14:41 2009
//
//--------------------------------------------------------------------


#include "patMath.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "bioArithExp.h"
#include "bioArithMult.h"
#include "bioArithCompositeLiteral.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

bioArithExp::bioArithExp(bioExpressionRepository* rep,
			 patULong par,
                         patULong left,
			 patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err) {


}

bioArithExp::~bioArithExp() {}

patString bioArithExp::getOperatorName() const {
  return ("exp") ;
}

patReal bioArithExp::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (child == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal();
    }
    if (child->isStructurallyZero()) {
      lastValue = 1.0;
      //return 1.0 ;
    } else {
      patReal childValue = child->getValue(prepareGradient, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal();
      }
      lastValue = patExp(childValue) ;
      //patReal result = patExp(childValue) ;
      //return result ;
    }
    lastComputedLap = currentLap;
  }

  return lastValue;
}


bioExpression* bioArithExp::getDerivative(patULong aLiteralId, patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
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

  bioExpression* e = new bioArithExp(theRepository, patBadId, value->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* result = new bioArithMult(theRepository,patBadId,deriv->getId(),e->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return result ;
}


bioArithExp* bioArithExp::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioExpression* leftClone(NULL) ;
  //bioExpression* rightClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
  }
  //if (rightChild != NULL) {
  //  rightClone = rightChild->getDeepCopy(err) ;
  //}

  bioArithExp* theNode = 
    new bioArithExp(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}

bioArithExp* bioArithExp::getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const {

  bioArithExp* theNode = 
    new bioArithExp(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return theNode ;
}



patString bioArithExp::getExpressionString() const {
  stringstream str ;
  str << "$E" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}



bioFunctionAndDerivatives* bioArithExp::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  bioFunctionAndDerivatives* c = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  result.theFunction =  patExp(c->theFunction) ;
  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
    if (c->theGradient[i] == 0.0) {
      result.theGradient[i] = 0.0 ;
    }
    else {
      result.theGradient[i] = result.theFunction * c->theGradient[i] ;
    }
  }
  if (result.theHessian != NULL && computeHessian) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (c->theGradient[i] == 0.0) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  patReal h = c->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  patReal v = result.theFunction * h ;
	  result.theHessian->setElement(i,j,v,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}
      }
      else {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  patReal h = c->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  patReal v = result.theFunction * (c->theGradient[i] * c->theGradient[j] + h) ;
	  result.theHessian->setElement(i,j,v,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}
      }
    }
  }
  
  return &result ;
  
}

