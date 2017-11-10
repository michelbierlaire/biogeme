//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteParameterProba.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDiscreteParameterProba.h"
#include "patProbaModel.h"
#include "patProbaPanelModel.h"
#include "patModelSpec.h"

patDiscreteParameterProba::patDiscreteParameterProba(patProbaModel* aModel,
						     patProbaPanelModel* aPanelModel) :
  model(aModel), panelModel(aPanelModel) {
  discreteParameters = patModelSpec::the()->getDiscreteParametersIterator() ;
  
  for (discreteParameters->first() ;
       !discreteParameters->isDone() ;
       discreteParameters->next()) {
    theDiscreteParameter = 
      discreteParameters->currentItem() ;
    DEBUG_MESSAGE("Parameter: " << theDiscreteParameter->name) ;
    if (!theDiscreteParameter->listOfTerms.empty()) {
      beginIterators.push_back(theDiscreteParameter->listOfTerms.begin());
      endIterators.push_back(theDiscreteParameter->listOfTerms.end());
      discreteParamId.push_back(theDiscreteParameter->theParameter->id) ;
    }
  }
  DELETE_PTR(discreteParameters) ;
  
  patError* err(NULL) ;
  theCombination = 
    new patGenerateCombinations<vector<patDiscreteTerm>::iterator,patDiscreteTerm >(beginIterators,endIterators,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
}

patReal patDiscreteParameterProba::evalProbaLog
(patObservationData* observation,
 patAggregateObservationData* aggObservation,
 patVariables* beta,
 const patVariables* parameters,
 patReal scale,
 unsigned long scaleIndex,
 patBoolean noDerivative,
 const vector<patBoolean>& compBetaDerivatives,
 const vector<patBoolean>& compParamDerivatives,
 patBoolean compMuDerivative,
 patBoolean compScaleDerivative,
 patVariables* betaDerivatives,
 patVariables* paramDerivatives,
 patReal* muDerivative,
 patReal* scaleDerivative,
 const patReal* mu,
 patSecondDerivatives* secondDeriv,
 patBoolean* success,
 patVariables* grad,
 patBoolean computeBHHH,
 vector<patVariables>* bhhh,
 patError*& err) {
  int debug = 0 ;
  patReal tmp(0.0) ;

  if (!noDerivative) {
    singleBetaDerivatives = patVariables(betaDerivatives->size(),0.0) ;
    singleParamDerivatives = patVariables(paramDerivatives->size(),0.0) ;
    singleMuDerivative = 0.0 ;
    singleScaleDeriv = 0.0 ;
  }


  for (theCombination->first() ;
       !theCombination->isDone() ;
       theCombination->next()) {  // loop on combinations
    ++debug ;
    oneCombi =  theCombination->currentItem() ;
    
    weightProbabilityOfTheCombination = 1.0 ;

    for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
	 i < oneCombi->size() ;
	 ++i) {
      
      patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
      
      patString name = patModelSpec::the()->getDiscreteParamName(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      patModelSpec::the()->setDiscreteParameterValue(name,
						     theTerm->massPoint,
						     err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      weightProbabilityOfTheCombination *= theTerm->probability->estimated ;
    }
    
    if (weightProbabilityOfTheCombination > 0) { // if nonzero proba

      patReal oneLogProba ;
      if (model != NULL) {
	oneLogProba = model->evalProbaLog(observation,
					  aggObservation,
					  beta,
					  parameters,
					  scale,
					  scaleIndex,
					  noDerivative,
					  compBetaDerivatives,
					  compParamDerivatives,
					  compMuDerivative,
					  compScaleDerivative,
					  &singleBetaDerivatives,
					  &singleParamDerivatives,
					  &singleMuDerivative,
					  &singleScaleDeriv,
					  mu,
					  secondDeriv,
					  success,
					  grad,
					  computeBHHH,
					  bhhh,
					  err) ;
      }
      else {
	err = new patErrMiscError("No model is specified") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      if (!(*success)) {
	return patReal() ;
      }
      
      
      patReal oneProba = exp(oneLogProba) ;
      patReal correctedProba = weightProbabilityOfTheCombination * oneProba ;
      
      tmp += correctedProba ;
      
      if (!noDerivative) { // noderivative
	
	
	// Affect the derivative with respect to the discretely
	// distributed parameter to the appropriate parameters
	
	for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
	     i < oneCombi->size() ;
	     ++i) {
	  patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
	  unsigned long actualBetaId = theTerm->massPoint->id ;
	  //	  patReal theDiscreteProba = theTerm->probability->estimated ;
	  unsigned long discreteId = discreteParamId[i] ;
	  
	  singleBetaDerivatives[actualBetaId] = singleBetaDerivatives[discreteId] ;
	  
	  if (compBetaDerivatives[theTerm->probability->id]) {
	    singleBetaDerivatives[theTerm->probability->id] = 1 / theTerm->probability->estimated ;
	  }
	}
	
	//Multiply all derivatives by the probability
	// Note that the derivatives are computed in the function for log(P). Therefore, they also have to be multiplied by P.
	
	
	for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    singleBetaDerivatives[i] *= correctedProba ;
	  }
	}
	for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    singleParamDerivatives[i] *= correctedProba ;
	  }
	}
	if (compMuDerivative) {
	  singleMuDerivative *= correctedProba ;
	}
	if (compScaleDerivative) {
	  singleScaleDeriv *= correctedProba ;
	}
	
	// Accumulate the derivatives
	
	
	for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    (*betaDerivatives)[i] += singleBetaDerivatives[i] ;
	    singleBetaDerivatives[i] = 0.0 ;
	  }
	}
	for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    (*paramDerivatives)[i] += singleParamDerivatives[i] ;
	    singleParamDerivatives[i] = 0.0 ;
	  }
	}
	if (compMuDerivative) {
	  (*muDerivative) += singleMuDerivative ;
	  singleMuDerivative = 0.0 ;
	}
	if (compScaleDerivative) {
	  (*scaleDerivative) += singleScaleDeriv ;
	  singleScaleDeriv = 0.0 ;
	}
      }
    }
  }
  if (!noDerivative) {
    for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
      if (compBetaDerivatives[i]) {
	(*betaDerivatives)[i] /= tmp ;
      }
    }
    for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
      if (compParamDerivatives[i]) {
	(*paramDerivatives)[i] /= tmp ;
      }
    }
    if (compMuDerivative) {
      (*muDerivative) /= tmp ;
    }
    if (compScaleDerivative) {
      (*scaleDerivative) /= tmp ;
    }
  }
  return(log(tmp)) ;
  
}


patReal patDiscreteParameterProba::evalPanelProba
(patIndividualData* observation,
   patVariables* beta,
   const patVariables* parameters,
   patReal scale,
   unsigned long scaleIndex,
   patBoolean noDerivative,
   const vector<patBoolean>& compBetaDerivatives,
   const vector<patBoolean>& compParamDerivatives,
   patBoolean compMuDerivative,
   patBoolean compScaleDerivative,
   patVariables* betaDerivatives,
   patVariables* paramDerivatives,
   patReal* muDerivative,
   patReal* scaleDerivative,
   const patReal* mu,
   patBoolean* success,
   patVariables* grad,
   patBoolean computeBHHH,
   vector<patVariables>* bhhh,
   patError*& err) {
  int debug = 0 ;
  patReal tmp(0.0) ;

  if (!noDerivative) {
    singleBetaDerivatives = patVariables(betaDerivatives->size(),0.0) ;
    singleParamDerivatives = patVariables(paramDerivatives->size(),0.0) ;
    singleMuDerivative = 0.0 ;
    singleScaleDeriv = 0.0 ;
  }


  for (theCombination->first() ;
       !theCombination->isDone() ;
       theCombination->next()) {  // loop on combinations
    ++debug ;
    oneCombi =  theCombination->currentItem() ;
    
    weightProbabilityOfTheCombination = 1.0 ;

    for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
	 i < oneCombi->size() ;
	 ++i) {
      
      patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
      
      patString name = patModelSpec::the()->getDiscreteParamName(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal();
      }
      
      patModelSpec::the()->setDiscreteParameterValue(name,
						     theTerm->massPoint,
						     err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      weightProbabilityOfTheCombination *= theTerm->probability->estimated ;
    }
    
    if (weightProbabilityOfTheCombination > 0) { // if nonzero proba

      patReal oneLogProba ;
      if (panelModel != NULL) {
	oneLogProba = panelModel->evalProbaLog(observation,
					       beta,
					       parameters,
					       scale,
					       scaleIndex,
					       noDerivative,
					       compBetaDerivatives,
					       compParamDerivatives,
					       compMuDerivative,
					       compScaleDerivative,
					       &singleBetaDerivatives,
					       &singleParamDerivatives,
					       &singleMuDerivative,
					       &singleScaleDeriv,
					       mu,
					       success,
					       grad,
					       computeBHHH,
					       bhhh,
					       err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal()  ;
	}
	if (!(*success)) {
	  return patReal() ;
	}
      
      }
      else {
	err = new patErrMiscError("No panel model is specified") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      patReal oneProba = exp(oneLogProba) ;
      patReal correctedProba = weightProbabilityOfTheCombination * oneProba ;
      
      tmp += correctedProba ;
      
      if (!noDerivative) { // noderivative
	
	
	// Affect the derivative with respect to the discretely
	// distributed parameter to the appropriate parameters
	
	for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
	     i < oneCombi->size() ;
	     ++i) {
	  patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
	  unsigned long actualBetaId = theTerm->massPoint->id ;
	  //	  patReal theDiscreteProba = theTerm->probability->estimated ;
	  unsigned long discreteId = discreteParamId[i] ;
	  
	  singleBetaDerivatives[actualBetaId] = singleBetaDerivatives[discreteId] ;
	  
	  if (compBetaDerivatives[theTerm->probability->id]) {
	    singleBetaDerivatives[theTerm->probability->id] = 1 / theTerm->probability->estimated ;
	  }
	}
	
	//Multiply all derivatives by the probability
	// Note that the derivatives are computed in the function for log(P). Therefore, they also have to be multiplied by P.
	
	
	for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    singleBetaDerivatives[i] *= correctedProba ;
	  }
	}
	for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    singleParamDerivatives[i] *= correctedProba ;
	  }
	}
	if (compMuDerivative) {
	  singleMuDerivative *= correctedProba ;
	}
	if (compScaleDerivative) {
	  singleScaleDeriv *= correctedProba ;
	}
	
	// Accumulate the derivatives
	
	
	for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    (*betaDerivatives)[i] += singleBetaDerivatives[i] ;
	    singleBetaDerivatives[i] = 0.0 ;
	  }
	}
	for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    (*paramDerivatives)[i] += singleParamDerivatives[i] ;
	    singleParamDerivatives[i] = 0.0 ;
	  }
	}
	if (compMuDerivative) {
	  (*muDerivative) += singleMuDerivative ;
	  singleMuDerivative = 0.0 ;
	}
	if (compScaleDerivative) {
	  (*scaleDerivative) += singleScaleDeriv ;
	  singleScaleDeriv = 0.0 ;
	}
      }
    }
  }
  if (!noDerivative) {
    for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
      if (compBetaDerivatives[i]) {
	(*betaDerivatives)[i] /= tmp ;
      }
    }
    for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
      if (compParamDerivatives[i]) {
	(*paramDerivatives)[i] /= tmp ;
      }
    }
    if (compMuDerivative) {
      (*muDerivative) /= tmp ;
    }
    if (compScaleDerivative) {
      (*scaleDerivative) /= tmp ;
    }
  }
  return(log(tmp)) ;
  
}

void patDiscreteParameterProba::generateCppCode(ostream& str,
					  patBoolean derivatives, 
					  patError*& err) {
  err = new patErrMiscError("Code generation not implemented for discrete parameters models") ;
  WARNING(err->describe()) ;
  return ;
}
