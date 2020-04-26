//-*-c++-*------------------------------------------------------------
//
// File name : bioArithProd.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed Jul  8 15:07:35  2009
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patError.h"
#include "bioParameters.h"

#include "bioArithProd.h"
#include "bioArithMult.h"
#include "bioArithDivide.h"
#include "bioArithSum.h"
#include "bioLiteralRepository.h"
#include "bioArithCompositeLiteral.h"
#include "bioRandomDraws.h"
#include "bioSample.h"
#include "bioRowIterator.h"
#include "bioMetaIterator.h"
#include "bioDrawIterator.h"

bioArithProd::bioArithProd(bioExpressionRepository* rep,
			   patULong par,
			   patULong left,
			   patString it,
			   patBoolean isPositive,
			   patError*& err) 
  : bioArithIterator(rep, par,left,it,err),
    allEntriesArePositive(isPositive), accessToFirstRow(patBadId)  {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
}


bioArithProd::~bioArithProd() {}


patString bioArithProd::getOperatorName() const {
  return patString("Prod") ;
}


bioExpression* bioArithProd::getDerivative(patULong aLiteralId, patError*& err) const {

  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* theProd = new bioArithProd(theRepository,
					    patBadId,
					    child->getId(),
					    theIteratorName,
					    allEntriesArePositive,
					    err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* leftValue = child->getDeepCopy(theRepository,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* leftResult = child->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* ratio = new bioArithDivide(theRepository,patBadId,leftResult->getId(),leftValue->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bioExpression* theSum = new bioArithSum(theRepository,patBadId,ratio->getId(),theIteratorName,patBadId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  bioExpression* result = new bioArithMult(theRepository,patBadId,theProd->getId(),theSum->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  return result ;
}


bioArithProd* bioArithProd::getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  bioArithProd* theNode = 
    new bioArithProd(rep,patBadId,leftClone->getId(),theIteratorName,allEntriesArePositive,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  return theNode ;
}

bioArithProd* bioArithProd::getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioArithProd* theNode = 
    new bioArithProd(rep,patBadId,child->getId(),theIteratorName,allEntriesArePositive,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  return theNode ;
}

patBoolean bioArithProd::isSum() const {
  return patFALSE ;
}

patBoolean bioArithProd::isProd() const {
  return patTRUE ;
}

patString bioArithProd::getExpressionString() const {
  stringstream str ;
  str << "$P" ;
  str << theIteratorName ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}


patReal bioArithProd::getValue(patBoolean prepareGradient,  patULong currentLap, patError*& err) {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (child->getId() != childId) {
      stringstream str ;
      str << "Ids are diffentent: " 
  	<< child->getId() 
  	<< " <> " 
  	<< childId ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
    
    if (theSample == NULL) {
      err = new patErrNullPointer("bioSample") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal result ;
    if (allEntriesArePositive) {
      result = 0.0 ;
    }
    else {
      result = 1.0 ;
    }

    if (theIteratorType == ROW) {
      bioRowIterator* theIter = theSample->createRowIterator(theCurrentSpan,
  							   theThreadSpan,
  							   patFALSE,
  							   err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      patULong nIter(0) ;
      if (allEntriesArePositive) {
        for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
        	++nIter ;
        	child->setVariables(theIter->currentItem()) ;
        	patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
        	if (err != NULL) {
        	  WARNING(err->describe()) ;
        	  return patReal() ;
        	}
        	result += log(expression) ;
        }
        result = exp(result) ;
      } else {
        for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
        	++nIter ;
        	child->setVariables(theIter->currentItem()) ;
        	patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
        	if (err != NULL) {
        	  WARNING(err->describe()) ;
        	  return patReal() ;
        	}
        	result *= expression ;
        }
      }
      DELETE_PTR(theIter) ;
      lastValue = result;
      //return result ;

    } else if (theIteratorType == META) {
      bioMetaIterator* theIter = theSample->createMetaIterator(theCurrentSpan,theThreadSpan,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      if (allEntriesArePositive) {
        for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
	  child->setCurrentSpan(theIter->currentItem()) ;
	  if (accessToFirstRow == patBadId) {
	    accessToFirstRow = bioParameters::the()->getValueInt("accessFirstDataFromMetaIterator",err) ;
	    if (accessToFirstRow) {
	      DEBUG_MESSAGE("Set variables") ;
	      child->setVariables(theIter->getFirstRow()) ;
	    }
	  }
	  patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }
          result += log(expression) ;
        }
        result = exp(result) ;
      } else {
        for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
        	child->setCurrentSpan(theIter->currentItem()) ;
		if (accessToFirstRow == patBadId) {
		  accessToFirstRow = bioParameters::the()->getValueInt("accessFirstDataFromMetaIterator",err) ;
		  if (accessToFirstRow) {
		    DEBUG_MESSAGE("Set variables") ;
		    child->setVariables(theIter->getFirstRow()) ;
		  }
		}
        	patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
        	if (err != NULL) {
        	  WARNING(err->describe()) ;
        	  return patReal() ;
        	}
          result *= expression ;
        }
      }
      DELETE_PTR(theIter) ;
      lastValue = result;
      //return result ;
    
    } else if (theIteratorType == DRAW) {
      err = new patErrMiscError("No product with draws") ;
      WARNING(err->describe()) ;
      return patReal() ;
    
    } else {
      err = new patErrMiscError("Should never be reached") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    lastComputedLap = result;
  }

  return lastValue;
}


patULong bioArithProd::getNumberOfOperations() const {
  patError* err(NULL) ;
    
  patULong result = 0 ;
    
    
  if (theIteratorType == ROW) {
    bioRowIterator* theIter = theSample->createRowIterator(theCurrentSpan,theThreadSpan,patFALSE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patULong nIter(0) ;
    for (theIter->first() ;
	 !theIter->isDone() ;
	 theIter->next()) {
      ++nIter ;
      result += child->getNumberOfOperations() ;
    }
    DELETE_PTR(theIter) ;
    return result ;
  }
  else if (theIteratorType == META) {
    bioMetaIterator* theIter = theSample->createMetaIterator(theCurrentSpan,theThreadSpan,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    for (theIter->first() ;
	 !theIter->isDone() ;
	 theIter->next()) {
      child->setCurrentSpan(theIter->currentItem()) ;
      result += child->getNumberOfOperations() ;
    }
    DELETE_PTR(theIter) ;
    return result ;
  }
  else if (theIteratorType == DRAW) {
    err = new patErrMiscError("Deprecated code.") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  err = new patErrMiscError("Should never be reached") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


bioFunctionAndDerivatives* bioArithProd::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian, patBoolean debugDeriv, patError*& err) {


  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }

  patReal r ;
  if (allEntriesArePositive) {
    r = 0.0 ;
  }
  else {
    r = 1.0 ;
  }

  
  vector<patReal> gradient(literalIds.size(),0.0) ;

  trHessian hessian (bioParameters::the()->getTrParameters(err),literalIds.size()) ;
  if (result.theHessian != NULL && computeHessian) {
    hessian.setToZero() ;
  }

  if (theIteratorType == ROW) {
    bioRowIterator* theIter = theSample->createRowIterator(theCurrentSpan,
							   theThreadSpan,
							   patFALSE,
							   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    patULong nIter(0) ;
    if (allEntriesArePositive) {
      for (theIter->first() ;
	   !theIter->isDone() ;
	   theIter->next()) {
	++nIter ;
	child->setVariables(theIter->currentItem()) ;
	bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	r += log(fg->theFunction) ;
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  if (fg->theGradient[i] != 0.0) {
	    if (performCheck) {
	      patReal v = fg->theGradient[i]/fg->theFunction ;
	      if (!isfinite(v)) {
		stringstream str ;
		str << "Result of " << fg->theGradient[i] << "/" << fg->theFunction << "=" << v ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      gradient[i] += v ;
	      if (!isfinite(gradient[i])) {
		stringstream str ;
		str << "Gradient[" << i << "]=" << gradient[i] ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	    }
	    else {
	      gradient[i] += fg->theGradient[i]/fg->theFunction ;
	    }
	  }
	}
	if (result.theHessian != NULL && computeHessian) {
	  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	    for (patULong j = i ; j < literalIds.size() ; ++j) {
	      patReal gs = fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (fg->theGradient[i] != 0 && fg->theGradient[j] != 0) {
		patReal v = 
		  (gs - fg->theGradient[i] * fg->theGradient[j] / fg->theFunction ) / fg->theFunction ;
		hessian.addElement(i,j,v,err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return NULL ;
		}
	      }
	      else {
		if (gs != 0) {
		  patReal v = 
		    gs  / fg->theFunction ;
		  hessian.addElement(i,j,v,err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return NULL ;
		  }
		}
	      }
	    }
	  }
	}
      }
      r = exp(r) ;
    }
    else {
      for (theIter->first() ;
	   !theIter->isDone() ;
	   theIter->next()) {
	++nIter ;
	child->setVariables(theIter->currentItem()) ;
	bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	r *= fg->theFunction ;
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  if (fg->theGradient[i] != 0.0) {
	    gradient[i] += fg->theGradient[i]/fg->theFunction ;
	  }
	}
	if (result.theHessian != NULL && computeHessian) {
	  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	    for (patULong j = i ; j < literalIds.size() ; ++j) {
	      patReal gs = fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (fg->theGradient[i] != 0 && fg->theGradient[j] != 0) {
		patReal v = 
		  (gs - fg->theGradient[i] * fg->theGradient[j] / fg->theFunction ) / fg->theFunction ;
		hessian.addElement(i,j,v,err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return NULL ;
		}
	      }
	      else {
		if (gs != 0) {
		  patReal v = 
		    gs  / fg->theFunction ;
		  if (!isfinite(v)) {
		    stringstream str ;
		    str << "Division: " << gs << " / " << fg->theFunction << " = " << v ;
		    err = new patErrMiscError(str.str()) ;
		    WARNING(err->describe()) ;
		    return NULL ;
		  }
		  hessian.addElement(i,j,v,err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return NULL ;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    DELETE_PTR(theIter) ;
    result.theFunction = r ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = r * gradient[i] ;
    }
    if (result.theHessian != NULL && computeHessian) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  patReal v = gradient[i] * result.theGradient[j] + result.theFunction * hessian.getElement(i,j,err) ;;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  result.theHessian->setElement(i,j,v,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}      
      }
    }
#ifdef  DEBUG
    patBoolean debugDeriv = (bioParameters::the()->getValueInt("debugDerivatives",err) != 0) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (debugDeriv != 0) {
      DEBUG_MESSAGE("Verify derivatives: " << getExpressionString()) ;
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
  else if (theIteratorType == META) {
    bioMetaIterator* theIter = theSample->createMetaIterator(theCurrentSpan,theThreadSpan,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (allEntriesArePositive) {
      for (theIter->first() ;
	   !theIter->isDone() ;
	   theIter->next()) {
	child->setCurrentSpan(theIter->currentItem()) ;
	bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	r += log(fg->theFunction) ;
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  if (fg->theGradient[i] != 0.0) {
	    gradient[i] += fg->theGradient[i] / fg->theFunction ;
	  }
	}
	if (result.theHessian != NULL && computeHessian) {
	  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	    for (patULong j = i ; j < literalIds.size() ; ++j) {
	      patReal gs = fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (fg->theGradient[i] != 0 && fg->theGradient[j] != 0) {
		patReal v = 
		  (gs - fg->theGradient[i] * fg->theGradient[j] / fg->theFunction ) / fg->theFunction ;
		hessian.addElement(i,j,v,err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return NULL ;
		}
	      }
	      else {
		if (gs != 0) {
		  patReal v = 
		    gs  / fg->theFunction ;
		  hessian.addElement(i,j,v,err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return NULL ;
		  }
		}
	      }
	    }
	  }
	}
      }
      r = exp(r) ;
    }
    else {
      for (theIter->first() ;
	   !theIter->isDone() ;
	   theIter->next()) {
	child->setCurrentSpan(theIter->currentItem()) ;
	bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	r *= fg->theFunction ;
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  if (fg->theGradient[i] != 0.0) {
	    gradient[i] += fg->theGradient[i] / fg->theFunction ;
	  }
	}
	if (result.theHessian != NULL && computeHessian) {
	  for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	    for (patULong j = i ; j < literalIds.size() ; ++j) {
	      patReal gs = fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (fg->theGradient[i] != 0.0 && fg->theGradient[j] != 0.0) {
		patReal v = 
		  (gs - fg->theGradient[i] * fg->theGradient[j] / fg->theFunction ) / fg->theFunction ;
		hessian.addElement(i,j,v,err) ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return NULL ;
		}
	      }
	      else {
		if (gs != 0.0) {
		  patReal v = 
		    gs  / fg->theFunction ;
		  hessian.addElement(i,j,v,err) ;
		  if (err != NULL) {
		    WARNING(err->describe()) ;
		    return NULL ;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    DELETE_PTR(theIter) ;
    result.theFunction = r ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = r * gradient[i] ;
    }    
    if (result.theHessian != NULL && computeHessian) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  patReal v = gradient[i] * result.theGradient[j] + result.theFunction * hessian.getElement(i,j,err) ;;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  result.theHessian->setElement(i,j,v,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}      
      }
    }
    
#ifdef  DEBUG
    patBoolean debugDeriv = (bioParameters::the()->getValueInt("debugDerivatives",err) != 0) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (debugDeriv != 0) {
      DEBUG_MESSAGE("Verify derivatives: " << getExpressionString()) ;
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
  else if (theIteratorType == DRAW) {
    err = new patErrMiscError("No product with draws") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  err = new patErrMiscError("Should never be reached") ;
  WARNING(err->describe()) ;
  return NULL ;
}
