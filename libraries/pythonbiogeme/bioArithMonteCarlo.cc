//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMonteCarlo.cc
// Author :    Michel Bierlaire
// Date :      Sun May 10 09:23:23 2015
//
//--------------------------------------------------------------------

#include <sstream>
#include <iomanip>

#include "bioReporting.h"
#include "patKalman.h"
#include "patStatistics.h"
#include "patFileNames.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patError.h"

#include "trParameters.h"

#include "bioDrawIterator.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h"
#include "bioRandomDraws.h"
#include "bioArithCompositeLiteral.h"

#include "bioSample.h"

#include "bioArithMonteCarlo.h"
#include "bioArithBinaryPlus.h"
#include "bioExpressionRepository.h"

/*!
*/
bioArithMonteCarlo::bioArithMonteCarlo(bioExpressionRepository* rep,
				       patULong par,
				       patULong left,
				       patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err), theIter(NULL), integrand(NULL), integral(NULL),theFilter(NULL), controlVariate(patFALSE) {
  
  numberOfDraws = bioParameters::the()->getValueInt("NbrOfDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (child->containsMonteCarlo()) {
    patULong allow = bioParameters::the()->getValueInt("allowNestedMonteCarlo",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (allow == 0) {
      err = new patErrMiscError("Nested MonteCarlo operators are not allowed. If you want to allow them, set the parameter 'allowNestedMonteCarlo' to 1") ;
      WARNING(err->describe()) ;
      return ;
    }
  }
}

bioArithMonteCarlo::bioArithMonteCarlo(bioExpressionRepository* rep,
				       patULong par,
				       patULong left,
				       patULong theIntegrand,
				       patULong theIntegral,
				       patError*& err) 
  : bioArithUnaryExpression(rep, par,left,err), theIter(NULL),theFilter(NULL), controlVariate(patTRUE) {

  if (child->containsMonteCarlo()) {
    patULong allow = bioParameters::the()->getValueInt("allowNestedMonteCarlo",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (allow == 0) {
      err = new patErrMiscError("Nested MonteCarlo operators are not allowed. If you want to allow them, set the parameter 'allowNestedMonteCarlo' to 1") ;
      WARNING(err->describe()) ;
      return ;
    }
  }

  numberOfDraws = bioParameters::the()->getValueInt("NbrOfDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  integrand = rep->getExpression(theIntegrand) ;
  if (integrand == NULL) {
    DEBUG_MESSAGE("Expression: " << getExpression(err)) ;
    stringstream str ;
    str << "No expression with id " << theIntegrand ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  relatedExpressions.push_back(integrand) ;
  integral = rep->getExpression(theIntegral) ;
  if (integral == NULL) {
    DEBUG_MESSAGE("Expression: " << getExpression(err)) ;
    stringstream str ;
    str << "No expression with id " << theIntegral ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  relatedExpressions.push_back(integral) ;

  patULong r = bioParameters::the()->getValueInt("monteCarloControlVariateReport",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  reportControlVariate = (r != 0) ;

  newDrawsDep.resize(numberOfDraws) ;
  newCvDrawsIndep.resize(numberOfDraws) ;

}

bioArithMonteCarlo::~bioArithMonteCarlo() {
  DEBUG_MESSAGE("release pointer") ;
  DELETE_PTR(theIter) ;
}

patString bioArithMonteCarlo::getOperatorName() const {
  return patString("MonteCarlo") ;
}

bioExpression* bioArithMonteCarlo::getDerivative(patULong aLiteralId, patError*& err) const {

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

  bioExpression* result = new bioArithMonteCarlo(theRepository,patBadId,leftResult->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}


bioArithMonteCarlo* bioArithMonteCarlo::getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (child != NULL) {
    leftClone = child->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  bioArithMonteCarlo* theNode = 
    new bioArithMonteCarlo(rep,patBadId,leftClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithMonteCarlo* bioArithMonteCarlo::getShallowCopy(bioExpressionRepository* rep,
						       patError*& err) const {
  bioArithMonteCarlo* theNode = 
    new bioArithMonteCarlo(rep,patBadId,child->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}



patString bioArithMonteCarlo::getExpressionString() const {
  stringstream str ;
  str << "$MC" ;
  if (child != NULL) {
    str << '{' << child->getExpressionString() << '}' ;
  }
  return patString(str.str()) ;
}


patReal bioArithMonteCarlo::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {

  patBoolean controlVariate = (integral != NULL) && (integrand != NULL) ;

  patReal result(0.0) ;
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

    if (controlVariate) {
      patReal controlVariateAnalytical = integral->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      calculateControlVariate(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      if (theFilter == NULL) {
	err = new patErrNullPointer("patKalman") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      result = theFilter->evaluate(controlVariateAnalytical) ;
    }
    else {
      if (theIter == NULL) {
	theIter = bioRandomDraws::the()->createIterator(err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
      }
      if (theIter == NULL) {
	err = new patErrNullPointer("bioDrawIterator") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }

      for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
	child->setDraws(theIter->currentItem()) ;
	// Calculate the current draw
	patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	result += expression ;
      }
      result /= numberOfDraws ;
    }
    lastValue = result;
    lastComputedLap = currentLap;
    return lastValue ;
  } else{
    return lastValue;
  }
}


patULong bioArithMonteCarlo::getNumberOfOperations() const {
  patError* err(NULL) ;
    
  patULong R = bioParameters::the()->getValueInt("NbrOfDraws",err)  ;
    
  return(1 + R * child->getNumberOfOperations()) ;
}


bioFunctionAndDerivatives* bioArithMonteCarlo::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }

  
  vector<patReal> gradient(literalIds.size(),0.0) ;
  if (result.theHessian != NULL && computeHessian) {
    result.theHessian->setToZero() ;
  }
  if (theIter == NULL) {
    theIter = bioRandomDraws::the()->createIterator(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  if (theIter == NULL) {
    err = new patErrNullPointer("bioDrawIterator") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  if (controlVariate) {
    bioFunctionAndDerivatives* theResult = 
      getNumericalFunctionAndGradientControlVariate(literalIds, computeHessian,err);
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    return theResult ;
  }

  patReal r(0.0) ;
  patReal R(numberOfDraws) ;
  for (theIter->first() ;
       !theIter->isDone() ;
       theIter->next()) {
    child->setDraws(theIter->currentItem()) ;
    bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (performCheck) {
      r += fg->theFunction / R;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	gradient[i] += fg->theGradient[i] / R ;
      }
      if (result.theHessian != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    patReal r = fg->theHessian->getElement(i,j,err) / patReal(R) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    result.theHessian->addElement(i,j,r,err) ;
	  }	
	}
      }
    }
    else {
      r += fg->theFunction ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	gradient[i] += fg->theGradient[i] ;
      }
      if (result.theHessian != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    patReal r = fg->theHessian->getElement(i,j,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    result.theHessian->addElement(i,j,r,err) ;
	  }	
	}
      }
    }
  }
  if (!performCheck) {
      r /= R ;
  }
  result.theFunction = r ;
  if (performCheck) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = gradient[i] ;
    }   
  }
  else {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = gradient[i]/patReal(R) ;
    }
    if (result.theHessian != NULL) {
      patReal invR = 1.0 / patReal(R) ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  result.theHessian->multElement(i,j,invR,err) ;
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




bioFunctionAndDerivatives* bioArithMonteCarlo::getNumericalFunctionAndGradientControlVariate(vector<patULong> literalIds, patBoolean computeHessian,patError*& err) {

  calculateControlVariate(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }


  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }

  patReal R(numberOfDraws) ;
  
  vector<patReal> gradient(literalIds.size(),0.0) ;
  if (result.theHessian != NULL && computeHessian) {
    result.theHessian->setToZero() ;
  }
  if (theIter == NULL) {
    theIter = bioRandomDraws::the()->createIterator(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  if (theIter == NULL) {
    err = new patErrNullPointer("bioDrawIterator") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  bioFunctionAndDerivatives* analytical = integral->getNumericalFunctionAndGradient(literalIds,computeHessian,patFALSE,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (theFilter == NULL) {
    err = new patErrNullPointer("patKalman") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  patReal coef = -theFilter->getCoefficient() ;
  
  result.theFunction = theFilter->evaluate(analytical->theFunction) ;
  for (theIter->first() ;
       !theIter->isDone() ;
       theIter->next()) {
    child->setDraws(theIter->currentItem()) ;
    bioFunctionAndDerivatives* fg = child->getNumericalFunctionAndGradient(literalIds,computeHessian,patFALSE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    integrand->setDraws(theIter->currentItem()) ;
    bioFunctionAndDerivatives* cv = integrand->getNumericalFunctionAndGradient(literalIds,computeHessian,patFALSE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    
    if (performCheck) {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	gradient[i] += (fg->theGradient[i] + coef * (cv->theGradient[i]-analytical->theGradient[i] )) / R ;
      }
      if (result.theHessian != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    patReal dd = fg->theHessian->getElement(i,j,err) / R ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    patReal cvd = cv->theHessian->getElement(i,j,err) / R ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    patReal anad = analytical->theHessian->getElement(i,j,err) / R ;
	    result.theHessian->addElement(i,j,dd+coef*(cvd-anad),err) ;
	  }	
	}
      }
    }
    else {
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	gradient[i] += (fg->theGradient[i] + coef * (cv->theGradient[i]-analytical->theGradient[i] )) ;
      }
      if (result.theHessian != NULL) {
	for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	  for (patULong j = i ; j < literalIds.size() ; ++j) {
	    patReal dd = fg->theHessian->getElement(i,j,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    patReal cvd = cv->theHessian->getElement(i,j,err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return NULL ;
	    }
	    patReal anad = analytical->theHessian->getElement(i,j,err) ;
	    result.theHessian->addElement(i,j,dd+coef*(cvd-anad),err) ;
	  }	
	}
      }
    }
  }

  if (performCheck) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = gradient[i] ;
    }   
  }
  else {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      result.theGradient[i] = gradient[i]/patReal(R) ;
    }
    if (result.theHessian != NULL) {
      patReal invR = 1.0 / patReal(R) ;
      for (patULong i = 0 ; i < literalIds.size() ; ++i) {
	for (patULong j = i ; j < literalIds.size() ; ++j) {
	  result.theHessian->multElement(i,j,invR,err) ;
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


void bioArithMonteCarlo::calculateControlVariate(patError*& err) {
  if (integral == NULL || integrand == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return ;
  }
  patReal controlVariateAnalytical = integral->getValue(patFALSE, patLapForceCompute, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }

  if (theIter == NULL) {
    theIter = bioRandomDraws::the()->createIterator(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
  }
  if (theIter == NULL) {
    err = new patErrNullPointer("bioDrawIterator") ;
    WARNING(err->describe()) ;
    return  ;
  }

  if (theFilter != NULL) {
    DELETE_PTR(theFilter) ;
  }

  patULong currentDraw(0) ;
  fill(newDrawsDep.begin(),newDrawsDep.end(),0.0) ;
  fill(newCvDrawsIndep.begin(),newCvDrawsIndep.end(),0.0) ;
  for (theIter->first() ; !theIter->isDone() ; theIter->next()) {
    child->setDraws(theIter->currentItem()) ;
    integrand->setDraws(theIter->currentItem()) ;
    // Calculate the current draw
    patReal expression = child->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    newDrawsDep[currentDraw] = expression ;

    // Calculate the control variate draw
    patReal cv = integrand->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    newCvDrawsIndep[currentDraw] = cv ;
    ++currentDraw ;
  }

  theFilter = new patKalman(newDrawsDep,newCvDrawsIndep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  
  if (reportControlVariate) {
    if (theReport == NULL) {
      err = new patErrNullPointer("bioReporting") ;
      WARNING(err->describe()) ;
      return ;
    }
    patStatistics theMainDraws ;
    patStatistics theCvDraws ;
    theMainDraws.addData(newDrawsDep) ;
    theCvDraws.addData(newCvDrawsIndep) ;
    theReport->addMonteCarloReport(&theMainDraws,
				   &theCvDraws,
				   theFilter,
				   controlVariateAnalytical,
				   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
  }
}

patBoolean bioArithMonteCarlo::containsMonteCarlo() const {
  return patTRUE ;
}

void bioArithMonteCarlo::checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) {
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->checkMonteCarlo(patTRUE,err) ;
    if (err != NULL) {
      WARNING("Expression: " << *this) ;
      WARNING(err->describe()) ;
      return ;
    }
  }
}
