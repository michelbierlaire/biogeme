//-*-c++-*------------------------------------------------------------
//
// File name : patGeneralizedUtility.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Mar  2 16:49:30 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patGeneralizedUtility.h"
#include "patModelSpec.h"
#include "patValueVariables.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"

patGeneralizedUtility::patGeneralizedUtility() {
  analyticalDerivatives = 
    patModelSpec::the()->utilDerivativesAvailableFromUser() ;
}

patGeneralizedUtility::~patGeneralizedUtility() {
}

patReal patGeneralizedUtility::computeFunction(unsigned long observationId,
					       unsigned long drawNumber,
					       unsigned long altId,
					       patVariables* beta,
					       vector<patObservationData::patAttributes>* x,
					       patError*& err)  {

  if (utilExpressions.empty()) {
    utilExpressions.resize(patModelSpec::the()->getNbrAlternatives(),NULL) ;
  }
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


  patValueVariables::the()->setVariables(beta) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (utilExpressions[altId] == NULL) {
    userId = patModelSpec::the()->getAltId(altId,err) ;
    utilExpressions[altId] = 
      patModelSpec::the()->getNonLinearUtilExpr(userId,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }

  if (utilExpressions[altId] == NULL) {
    return 0.0 ;
  }
  
  result = utilExpressions[altId]->getValue(err) ;
  if (err != NULL) {
    WARNING("Error in evaluating expression " << *expression) ;
    WARNING(err->describe()) ;
    return patReal() ;    
  }
  return result ;
}
  

patVariables* patGeneralizedUtility::
computeBetaDerivative(unsigned long observationId,
		      unsigned long drawNumber,
		      unsigned long altId,
		      patVariables* beta,
		      vector<patObservationData::patAttributes>* x,
		      patVariables* betaDerivatives,
		      patError*& err)  {

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

  if (utilExpressions.empty()) {
    utilExpressions.resize(patModelSpec::the()->getNbrAlternatives(),NULL) ;
  }
  if (analyticalDerivatives) {
    if (derivExpressions.empty()) {
      derivExpressions.resize(patModelSpec::the()->getNbrAlternatives(),
			      vector<patArithNode*>(beta->size(),NULL)) ;
      derivFirst.resize(patModelSpec::the()->getNbrAlternatives(),
			      vector<patBoolean>(beta->size(),patTRUE)) ;
    }
  }

  patValueVariables::the()->setVariables(beta) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return betaDerivatives ;
  }

  if (utilExpressions[altId] == NULL) {
    userId = patModelSpec::the()->getAltId(altId,err) ;
    utilExpressions[altId] = patModelSpec::the()->getNonLinearUtilExpr(userId,
								       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return betaDerivatives ;
    }
  }

  if (utilExpressions[altId] == NULL) {
    return betaDerivatives ;
  }


  for (i = 0 ; i < beta->size() ; ++i) {

    if (analyticalDerivatives) {
      if (derivFirst[altId][i]) {
	derivFirst[altId][i] = patFALSE ;
	patArithNode* tmp = patModelSpec::the()->getDerivative(altId,i,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return betaDerivatives ;    
	}
	if (tmp != NULL) {
	  tmp->computeParamId(err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return betaDerivatives ;    
	  }
	  derivExpressions[altId][i] = tmp ;
	}
      }
    }
    
    if (analyticalDerivatives) {
      if (derivExpressions[altId][i] == NULL) {
	(*betaDerivatives)[i] = 0.0 ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return betaDerivatives ;    
	}
      }
      else {
	(*betaDerivatives)[i] = derivExpressions[altId][i]->getValue(err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return betaDerivatives ;    
	}
      }
    }
    else {
      (*betaDerivatives)[i] = utilExpressions[altId]->getDerivative(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return betaDerivatives ;    
      }
    }
    if (!isfinite((*betaDerivatives)[i])) {
      stringstream str ;
      str << "Numerical problem when computing derivative " << i << ": " << (*betaDerivatives)[i] ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return betaDerivatives;
    }
    
  }
  return betaDerivatives ;
    
}


patVariables patGeneralizedUtility::
computeCharDerivative(unsigned long observationId,
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

  err = new patErrMiscError("Not yet implemented...") ;
  WARNING(err->describe()) ;
  return patVariables() ;

}


patString patGeneralizedUtility::getName() const {
  return patString("Nonlinear stochastic utility function") ;
} ;


// void patGeneralizedUtility::assignDrawsValues(unsigned long observationId,
// 					      unsigned long drawNumber,
// 					      patError*& err) const {
//   if (theDraws == NULL) {
//     err = new patErrNullPointer("patRandomDraws") ;
//     WARNING(err->describe()) ;
//     return ;
//   }

//   patIterator<patString>* iter = 
//     patModelSpec::the()->createDrawNamesIterator() ;
//   for (iter->first() ;
//        !iter->isDone() ;
//        iter->next()) {
//     patString drawName = iter->currentItem() ;
//     patReal value = theDraws->getDraw(drawName,drawNumber,observationId,err) ;
//     if (err != NULL) {
//       WARNING(err->describe()) ;
//       return ;
//     }
//     patValueVariables::the()->setValue(drawName,value) ;
//   }
// }

void patGeneralizedUtility::generateCppCode(ostream& cppFile,
					    unsigned long altId,
					    patError*& err) {
  cppFile << "    //////////////////////////////////" << endl ;
  cppFile << "    // Code generated in patGeneralizedUtility" << endl ;
  if (utilExpressions.empty()) {
    utilExpressions.resize(patModelSpec::the()->getNbrAlternatives(),NULL) ;
  }

  if (utilExpressions[altId] == NULL) {
    userId = patModelSpec::the()->getAltId(altId,err) ;
    utilExpressions[altId] = 
      patModelSpec::the()->getNonLinearUtilExpr(userId,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  
  if (utilExpressions[altId] == NULL) {
    cppFile << "0.0 ;" << endl ;
    cppFile << "    // end of code generated in patGeneralizedUtility" << endl ;
    cppFile << "    /////////////////////////////////////////////////" << endl ;
    return ; 
  }
  
  patString cppCode = utilExpressions[altId]->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  cppFile << cppCode << " ; " << endl ;
  cppFile << "    // end of code generated in patGeneralizedUtility" << endl ;
  cppFile << "    /////////////////////////////////////////////////" << endl ;
  
  
}

void patGeneralizedUtility::generateCppDerivativeCode(ostream& cppFile,
						      unsigned long altId,
						      unsigned long betaId,
						      patError*& err) {


  cppFile << "////////////////////////////////" << endl ;
  cppFile << "/// Code in patGeneralizedUtility" << endl ;
  if (utilExpressions.empty()) {
    utilExpressions.resize(patModelSpec::the()->getNbrAlternatives(),NULL) ;
  }
  if (analyticalDerivatives) {
    if (derivExpressions.empty()) {
      derivExpressions.resize(patModelSpec::the()->getNbrAlternatives(),
			      vector<patArithNode*>(patModelSpec::the()->getNbrNonFixedParameters(),NULL)) ;
      derivFirst.resize(patModelSpec::the()->getNbrAlternatives(),
			vector<patBoolean>(patModelSpec::the()->getNbrNonFixedParameters(),patTRUE)) ;
    }
  }

  
  if (utilExpressions[altId] == NULL) {
    userId = patModelSpec::the()->getAltId(altId,err) ;
    utilExpressions[altId] = patModelSpec::the()->getNonLinearUtilExpr(userId,
								       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
  }
  
  if (utilExpressions[altId] == NULL) {
    cppFile << "0.0" << endl ;
    return ;
  }


  if (analyticalDerivatives) {
    if (derivFirst[altId][betaId]) {
      derivFirst[altId][betaId] = patFALSE ;
      patArithNode* tmp = patModelSpec::the()->getDerivative(altId,betaId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;    
      }
      if (tmp != NULL) {
	tmp->computeParamId(err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return  ;    
	}
	derivExpressions[altId][betaId] = tmp ;
      }
    }
  }
    
  if (analyticalDerivatives) {
    if (derivExpressions[altId][betaId] == NULL) {
	cppFile << "0.0" << endl ;
    }
    else {
      cppFile <<  derivExpressions[altId][betaId]->getCppCode(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;    
      }
    }
  }
  else {
    patString theCode = utilExpressions[altId]->getCppDerivativeCode(betaId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;    
    }
    cppFile << theCode << endl ;
  }
  cppFile << "/// end of code in patGeneralizedUtility" << endl ;
  cppFile << "////////////////////////////////" << endl ;
}

