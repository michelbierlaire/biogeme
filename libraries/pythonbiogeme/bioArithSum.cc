//-*-c++-*------------------------------------------------------------
//
// File name : bioArithSum.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jun 16 12:47:18  2009
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
#include "patError.h"

#include "trHessian.h"
#include "trParameters.h"

#include "bioRowIterator.h"
#include "bioMetaIterator.h"
#include "bioDrawIterator.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h"
#include "bioRandomDraws.h"
#include "bioArithCompositeLiteral.h"

#include "bioIteratorInfoRepository.h"
#include "bioSample.h"

#include "bioArithSum.h"
#include "bioArithBinaryPlus.h"
#include "bioLiteralRepository.h"
#include "bioExpressionRepository.h"

/*!
*/
bioArithSum::bioArithSum(bioExpressionRepository* rep,
			 patULong par,
                         patULong left,
                         patString it,
			 patULong w,
			 patError*& err) 
  : bioArithIterator(rep, par,left,it,err),bhhh(NULL),weightId(w),accessToFirstRow(patBadId) {

  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  
  weight = theRepository->getExpression(w) ;

  


}

bioArithSum::~bioArithSum() {}

patString bioArithSum::getOperatorName() const {
  return patString("Sum") ;
}

bioExpression* bioArithSum::getDerivative(patULong aLiteralId, patError*& err) const {

  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  bioExpression* leftResult = child->getDerivative(aLiteralId,err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* result = new bioArithSum(theRepository,patBadId,leftResult->getId(),theIteratorName,weightId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}


bioArithSum* bioArithSum::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  bioArithSum* theNode = 
    new bioArithSum(rep,patBadId,leftClone->getId(),theIteratorName,weightId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithSum* bioArithSum::getShallowCopy(bioExpressionRepository* rep,
					 patError*& err) const {
  bioArithSum* theNode = 
    new bioArithSum(rep,patBadId,child->getId(),theIteratorName,weightId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patBoolean bioArithSum::isSum() const {
  return patTRUE ;
}

patBoolean bioArithSum::isProd() const {
  return patFALSE ;
}

patString bioArithSum::getExpressionString() const {
  stringstream str ;
  str << "$S" ;
  str << theIteratorName ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}

patBoolean bioArithSum::isSumIterator() const {
  return patTRUE ;
}




patReal bioArithSum::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {

  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    if (child->getId() != childId) {
      stringstream str ;
      str << "Ids are different: " 
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
    patReal result = 0.0 ;
    
    if(bhhh != NULL) {
      bhhh->setToZero() ;
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
      for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
        ++nIter ;
        child->setVariables(theIter->currentItem()) ;
        patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
        if (err != NULL) {
        	WARNING(err->describe()) ;
        	return patReal() ;
        }
	
        if (weight != NULL && isTop()) {
        	weight->setVariables(theIter->currentItem()) ;
        	patReal w = weight->getValue(patFALSE, patLapForceCompute, err) ;
        	if (err != NULL) {
        	  WARNING(err->describe()) ;
        	  return patReal() ;
        	}
          result += w * expression ;
        } else {
          result += expression ;
        }
      }

      DELETE_PTR(theIter) ;
      //return result ;
    
    } else if (theIteratorType == META) {
      bioMetaIterator* theIter = theSample->createMetaIterator(theCurrentSpan,theThreadSpan,err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
        child->setCurrentSpan(theIter->currentItem()) ;
	if (accessToFirstRow == patBadId) {
	  accessToFirstRow = bioParameters::the()->getValueInt("accessFirstDataFromMetaIterator",err) ;
	  if (accessToFirstRow) {
	    child->setVariables(theIter->getFirstRow()) ;
	  }
	}
        patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
        if (err != NULL) {
        	WARNING(err->describe()) ;
        	return patReal() ;
        }

        result += expression ;
      }
      DELETE_PTR(theIter) ;
      //return result ;
    
    } else if (theIteratorType == DRAW) {
      err = new patErrMiscError("Deprecated code.") ;
      WARNING(err->describe()) ;
      return patReal() ;
      
    }else{
      err = new patErrMiscError("Should never be reached") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    lastValue = result;
    lastComputedLap = currentLap;
    return lastValue ;
  }else{
    return lastValue;
  }

}


trHessian* bioArithSum::getBhhh() {
  return bhhh ;
}


patULong bioArithSum::getNumberOfOperations() const {
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


bioFunctionAndDerivatives* bioArithSum::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {

  patBoolean scaleDerivatives = (bioParameters::the()->getValueInt("scaleDerivativesInSums") != 0) ;
  
  patReal theScale(0.0) ;
  if (scaleDerivatives) {
    theScale = patMax(patReal(1.0),patAbs(getValue(patFALSE,patLapForceCompute,err))) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  if (isTop() && bhhh == NULL) {
    trParameters p = bioParameters::the()->getTrParameters(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    bhhh = new trHessian(p,literalIds.size()) ;
  }


  
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  patReal rfct(0.0) ;
  
  vector<patReal> gradient(literalIds.size(),0.0) ;
  if (result.theHessian != NULL && computeHessian) {
    result.theHessian->setToZero() ;
  }

  if(bhhh != NULL) {
    bhhh->setToZero() ;
  }
  
  if (theIteratorType == ROW) {
    bioRowIterator* theIter = theSample->createRowIterator(theCurrentSpan,
							   theThreadSpan,
							   patFALSE,
							   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    patULong nIter(0) ;
    for (theIter->first() ;
	 !theIter->isDone() ;
	 theIter->next()) {
      ++nIter ;
      child->setVariables(theIter->currentItem()) ;
      bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
      if (err != NULL) {
	patULong theRow = theIter->getCurrentRow() ;
	stringstream str ;
	str << "Row " << theRow << ": " << err->describe() ; 
	DELETE_PTR(err) ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return NULL ;
      }
      patReal w ;
      if (weight != NULL && isTop()) {
	weight->setVariables(theIter->currentItem()) ;
	w = weight->getValue(patFALSE, patLapForceCompute, err) ;
      }
      if ( rfct < patMaxReal) {
	if (fg->theFunction == patMaxReal) {
	  rfct = patMaxReal ;
	}
	else {
	  if (weight != NULL && isTop()) {
	    rfct += w * fg->theFunction ;
	  }
	  else {
	    rfct += fg->theFunction ;
	  }
	}
      }
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	if (weight != NULL && isTop()) {
	  patReal term(w * fg->theGradient[i]) ;
	  if (scaleDerivatives && theScale != 1.0) {
	    term /= theScale ;
	  }
	  gradient[i] += term;
	}
	else {
	  patReal term(fg->theGradient[i]) ;
	  if (scaleDerivatives && theScale != 1.0) {
	    term /= theScale ;
	  }
	  gradient[i] += term ;
	}
      }
      if (result.theHessian != NULL && computeHessian) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    if (weight != NULL && isTop()) {
	      patReal elem = w * fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		patULong theRow = theIter->getCurrentRow() ;
		stringstream str ;
		str << "Row " << theRow << ": " << err->describe() ; 
		DELETE_PTR(err) ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (scaleDerivatives && theScale != 1.0) {
		elem /= theScale ;
	      }
	      result.theHessian->addElement(i,j,elem,err) ;
	      if (err != NULL) {
		patULong theRow = theIter->getCurrentRow() ;
		stringstream str ;
		str << "Row " << theRow << ": " << err->describe() ; 
		DELETE_PTR(err) ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	    }
	    else {
	      patReal elem = fg->theHessian->getElement(i,j,err) ;
	      if (err != NULL) {
		patULong theRow = theIter->getCurrentRow() ;
		stringstream str ;
		str << "Row " << theRow << ": " << err->describe() ; 
		DELETE_PTR(err) ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	      if (scaleDerivatives && theScale != 1.0) {
		elem /= theScale ;
	      }
	      result.theHessian->addElement(i,j,elem,err) ;
	      if (err != NULL) {
		patULong theRow = theIter->getCurrentRow() ;
		stringstream str ;
		str << "Row " << theRow << ": " << err->describe() ; 
		DELETE_PTR(err) ;
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
		return NULL ;
	      }
	    }
	  }
	}
      }
      if (bhhh != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    if (weight != NULL && isTop()) {

	      // This is the reason why the weight has to be handled
	      // separately. It only needs to be multipled once for
	      // the computation of the BHHH.
	      bhhh->addElement(i,j,w * fg->theGradient[i]*fg->theGradient[j],err) ;
	    }
	    else {
	      bhhh->addElement(i,j,fg->theGradient[i]*fg->theGradient[j],err) ;
	    }
	    if (err != NULL) {
	      patULong theRow = theIter->getCurrentRow() ;
	      stringstream str ;
	      str << "Row " << theRow << ": " << err->describe() ; 
	      DELETE_PTR(err) ;
	      err = new patErrMiscError(str.str()) ;
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	  }
	}
      }
    }
    DELETE_PTR(theIter) ;


    result.theFunction = rfct ;
    if (scaleDerivatives && theScale != 1.0) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = gradient[i] * theScale ;
      }
      result.theHessian->multAllEntries(theScale,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
    else {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = gradient[i] ;
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
  else if (theIteratorType == META) {
    bioMetaIterator* theIter = theSample->createMetaIterator(theCurrentSpan,theThreadSpan,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    for (theIter->first() ;
	 !theIter->isDone() ;
	 theIter->next()) {
      child->setCurrentSpan(theIter->currentItem()) ;
      bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      rfct += fg->theFunction ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	patReal term(fg->theGradient[i]) ;
	if (scaleDerivatives && theScale != 1.0) {
	  term /= theScale ;
	}
	gradient[i] += term ;
      }
      if (result.theHessian != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    patReal r = fg->theHessian->getElement(i,j,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    if (scaleDerivatives && theScale != 1.0) {
	      r /= theScale ;
	    }
	    result.theHessian->addElement(i,j,r,err) ;
	  }	
	}
      }
      if (bhhh != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    bhhh->addElement(i,j,fg->theGradient[i]*fg->theGradient[j],err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	  }
	}
      }
    }
    DELETE_PTR(theIter) ;
    result.theFunction = rfct ;
    if (scaleDerivatives && theScale != 1.0) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = gradient[i] * theScale ;
      }
      result.theHessian->multAllEntries(theScale,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
    else {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	result.theGradient[i] = gradient[i] ;
      }    
    }
#ifdef  DEBUG
    patBoolean debugDeriv = (bioParameters::the()->getValueInt("debugDerivatives",err) != 0) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
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
  else if (theIteratorType == DRAW) {
    err = new patErrMiscError("Deprecated code.") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  err = new patErrMiscError("Should never be reached") ;
  WARNING(err->describe()) ;
  return NULL ;
}




