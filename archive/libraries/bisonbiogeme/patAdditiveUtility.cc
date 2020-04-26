//-*-c++-*------------------------------------------------------------
//
// File name : patAdditiveUtility.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Mar  3 17:55:53 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patAdditiveUtility.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

void patAdditiveUtility::addUtility(patUtility* aUtil) {
  listOfUtilities.push_back(aUtil) ;
}


patReal patAdditiveUtility::computeFunction(unsigned long observationId,
					    unsigned long drawNumber,
					    unsigned long altId,
					    patVariables* beta,
					    vector<patObservationData::patAttributes>* x,
					    patError*& err) {
  
  if (beta == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  if (x == NULL) {
    err = new patErrNullPointer("vector<patObservationData::patAttributes>") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal result(0.0) ;

  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (*i == NULL) {
      err = new patErrNullPointer("patUtility") ;
      WARNING(err->describe());
      return patReal() ;
    }
    result += (*i)->computeFunction(observationId,drawNumber,altId,beta,x,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  return result ;
}
  

patVariables* patAdditiveUtility::
computeBetaDerivative(unsigned long observationId,
		      unsigned long drawNumber,
		      unsigned long altId,
		      patVariables* beta,
		      vector<patObservationData::patAttributes>* x,
		      patVariables* betaDerivatives,
		      patError*& err) {

  if (betaDerivatives == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }

  if (beta == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }
  if (x == NULL) {
    err = new patErrNullPointer("vector<patObservationData::patAttributes>") ;
    WARNING(err->describe()) ;
    return betaDerivatives ;
  }
  if (betaDerivatives->size() != beta->size()) {
    stringstream str ;
    str << "Size of betaDerivatives should be " << beta->size() 
	<< " and not " << betaDerivatives->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return betaDerivatives  ;
  }

  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (*i == NULL) {
      err = new patErrNullPointer("patUtility") ;
      WARNING(err->describe());
      return betaDerivatives ;
    }
    patVariables tmp(beta->size()) ;
    (*i)->computeBetaDerivative(observationId,
				drawNumber,
				altId,
				beta,
				x,
				&tmp,
				err) ;
    (*betaDerivatives) += tmp ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
  }
  return betaDerivatives ;
}

patVariables patAdditiveUtility::computeCharDerivative(unsigned long observationId,
						       unsigned long drawNumber,
						       unsigned long altId,
						       patVariables* beta,
						       vector<patObservationData::patAttributes>* x,
						       patError*& err)  {

  if (beta == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patVariables()  ;
  }
  if (x == NULL) {
    err = new patErrNullPointer("vector<patObservationData::patAttributes>") ;
    WARNING(err->describe()) ;
    return patVariables() ;
  }
  patVariables result(beta->size(),0.0) ;

  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (*i == NULL) {
      err = new patErrNullPointer("patUtility") ;
      WARNING(err->describe());
      return patVariables() ;
    }
    result += (*i)->computeCharDerivative(observationId,drawNumber,altId,beta,x,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patVariables() ;
    }
  }
  return result ;
}

patString patAdditiveUtility::getName() const {
  stringstream str ;
  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (*i != NULL) {
      if (i != listOfUtilities.begin()) {
	str << "+" ;
      }
      str << (*i)->getName() ;
    }
  }
  return patString(str.str()) ;
}

void patAdditiveUtility::generateCppCode(ostream& cppFile,
					 unsigned long altId,
					 patError*& err) {

  patBoolean firstTerm = patTRUE ;
  
  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (!firstTerm) {
      cppFile << " + "  << endl;
    }
    if (*i == NULL) {
      err = new patErrNullPointer("patUtility") ;
      WARNING(err->describe());
      return  ;
    }
    (*i)->generateCppCode(cppFile,altId,err) ;
    firstTerm = patFALSE ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}

void patAdditiveUtility::generateCppDerivativeCode(ostream& cppFile,
						   unsigned long altId,
						   unsigned long betaId,
						   patError*& err) {

  patBoolean firstTerm = patTRUE ;
  
  for (vector<patUtility*>::const_iterator i = listOfUtilities.begin() ;
       i != listOfUtilities.end() ;
       ++i) {
    if (!firstTerm) {
      cppFile << " + " << endl ;
    }
    if (*i == NULL) {
      err = new patErrNullPointer("patUtility") ;
      WARNING(err->describe());
      return  ;
    }
    (*i)->generateCppDerivativeCode(cppFile,altId,betaId,err) ;
    firstTerm = patFALSE ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

}
