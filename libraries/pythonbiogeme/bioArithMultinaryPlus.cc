//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMultinaryPlus.cc
// Author :    Michel Bierlaire
// Date :      Mon May 23 17:58:41 2011
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef DEBUG
#include "bioParameters.h"
#endif
#include "bioArithMultinaryPlus.h"
#include "patDisplay.h"
#include "patErrMiscError.h"

bioArithMultinaryPlus::bioArithMultinaryPlus(bioExpressionRepository* rep,
					     patULong par,
					     vector<patULong> l,
					     patError*& err) :
  bioArithMultinaryExpression(rep,par,l,err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

bioArithMultinaryPlus::~bioArithMultinaryPlus() {

}

patString bioArithMultinaryPlus::getOperatorName() const {
  return patString("Add") ;
}

patReal bioArithMultinaryPlus::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    patReal result(0.0) ;
    for (vector<bioExpression*>::iterator i = listOfChildren.begin() ;
         i != listOfChildren.end() ;
         ++i) {
      patReal r = (*i)->getValue(prepareGradient, currentLap, err);
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal();
      }
      result += r ;
    }
    
    lastValue =  result ;
    lastComputedLap = currentLap;
  }
  return lastValue;
}

bioExpression* bioArithMultinaryPlus::getDerivative(patULong aLiteralId, 
						    patError*& err) const {
  vector<patULong> d ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    bioExpression* deriv = (*i)->getDerivative(aLiteralId,err);
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    d.push_back(deriv->getId()) ;
  }
  bioArithMultinaryPlus* result = new bioArithMultinaryPlus(theRepository,patBadId,d,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return result ;
}

bioArithMultinaryPlus* bioArithMultinaryPlus::getDeepCopy(bioExpressionRepository* rep,
							  patError*& err) const {
  vector<patULong> d ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    bioExpression* c = (*i)->getDeepCopy(rep,err);
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    d.push_back(c->getId()) ;
  }
  bioArithMultinaryPlus* result = new bioArithMultinaryPlus(rep,patBadId,d,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return result ;

}

bioArithMultinaryPlus* bioArithMultinaryPlus::getShallowCopy(bioExpressionRepository* rep,
							  patError*& err) const {
  bioArithMultinaryPlus* result = new bioArithMultinaryPlus(rep,patBadId,listOfChildrenIds,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return result ;

}

patString bioArithMultinaryPlus::getExpression(patError*& err) const {
  stringstream str ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if (i != listOfChildren.begin()) {
      str << " + " ;
    }
    str << (*i)->getExpression(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
  }
  return patString(str.str()) ;
  
}

patString bioArithMultinaryPlus::getExpressionString() const {
  stringstream str ;
  str << "$Add{" ;
  for (vector<bioExpression*>::const_iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    if (i != listOfChildren.begin()) {
      str << "," ;
    }
    str << (*i)->getExpressionString() ;
  }
  str << "}" ;
  return patString(str.str()) ;

}


bioFunctionAndDerivatives* bioArithMultinaryPlus::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  result.theFunction = 0.0 ;
  for (patULong k = 0 ; k < literalIds.size() ; ++k) {
    result.theGradient[k] = 0.0 ;
  }
  if (result.theHessian != NULL && computeHessian) {
    result.theHessian->setToZero() ;
  }
  for (vector<bioExpression*>::iterator i = listOfChildren.begin() ;
       i != listOfChildren.end() ;
       ++i) {
    bioFunctionAndDerivatives* fg = (*i)->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    result.theFunction += fg->theFunction ;
    for (patULong k = 0 ; k < literalIds.size() ; ++k) {
      result.theGradient[k] += fg->theGradient[k] ;
    }
    if (result.theHessian != NULL && computeHessian) {
      for (patULong k = 0 ; k < literalIds.size() ; ++k) {
	for (patULong j = k ; j < literalIds.size() ; ++j) {
	  patReal v = fg->theHessian->getElement(k,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  result.theHessian->addElement(k,j,v,err) ;
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
