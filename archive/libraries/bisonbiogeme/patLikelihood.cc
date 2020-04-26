//-*-c++-*------------------------------------------------------------
//
// File name : patLikelihood.cc
// Author :    Michel Bierlaire
// Date :      Tue Aug  8 23:13:39 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <numeric>
#include <sstream>
#include <cassert>

#include "patConst.h"
#include "patMath.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"
#include "patGEV.h"
#include "patLikelihood.h"
#include "patProbaModel.h"
#include "patProbaPanelModel.h"
#include "patUtility.h"
#include "patSample.h" 
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patValueVariables.h"
#include "patTimer.h"
#include "patSecondDerivatives.h"
#include "patDiscreteParameterProba.h"

patLikelihood::patLikelihood() :  model(NULL), sample(NULL), scaleIndex(patBadId) , theAggObsIterator(NULL), theObsIterator(NULL), theIndIterator(NULL)  {}
patLikelihood::patLikelihood(patProbaModel* aModel, 
			     patProbaPanelModel* aPanelModel,
			     patSample* aSample,
			     patError*& err) :
  model(aModel), panelModel(aPanelModel), sample(aSample),theDiscreteParamModel(NULL) , theAggObsIterator(NULL), theObsIterator(NULL), theIndIterator(NULL)  {

  areObservationsAggregate = patModelSpec::the()->isAggregateObserved() ;
  if (areObservationsAggregate) {
    theAggObsIterator = aSample->createAggObsIterator() ;
  }

  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
    
    theDiscreteParamModel = new patDiscreteParameterProba(aModel,aPanelModel) ;
  }
}

patLikelihood::~patLikelihood() {
  
}

void patLikelihood::setModel(patProbaModel* aModel) {
  model = aModel ;
}

void patLikelihood::setModel(patProbaPanelModel* aPanelModel) {
  panelModel = aPanelModel ;
}

patProbaModel* patLikelihood::getModel() {
  return model ;
}

patProbaPanelModel* patLikelihood::getPanelModel() {
  return panelModel ;
}

void patLikelihood::setSample(patSample* aSample) {
  sample = aSample ;
}

patSample* patLikelihood::getSample() {
  return sample ;
}

patReal patLikelihood::evaluate(patVariables* betaParameters,
				const patReal* mu,
				const patVariables* modelParameters,
				const patVariables* scaleParameters,
				patBoolean* success,
				patVariables* grad,
				patError*& err) {
  
  //Evaluate the likelihood function for given values of beta parameters
  //(involved in the utilities), model parameters (like mu_m parameters) and
  //scale parameters


  if (err != NULL) {
    WARNING(err->describe()) ;
    return  patReal() ;
  }

  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (betaParameters == NULL || modelParameters == NULL || scaleParameters == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (mu == NULL ) {
    err = new patErrNullPointer("patReal") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  vector<patBoolean> dummy ;

  patReal res = evaluateLog(betaParameters,
			    mu,
			    modelParameters,
			    scaleParameters,
			    patTRUE,
			    dummy,
			    dummy,
			    patFALSE,
			    dummy,
			    NULL,
			    NULL,
			    NULL,
			    NULL,
			    NULL,
			    success,
			    grad,
			    patFALSE,
			    NULL,
			    NULL,
			    err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (!(*success)) {
    return patMaxReal ;
  }

  if (res > 0) {
    err = new patErrMiscError("Value should be negative") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (res >= patLogMaxReal::the()) {
    WARNING("Overflow: exp(" << res << ")") ;
  }

  patReal outputVariable = exp(res) ;
  return outputVariable ;


}
  
patReal patLikelihood::evaluateLog(patVariables* betaParameters,
				   const patReal* mu,
				   const patVariables* modelParameters,
				   const patVariables* scaleParameters,
				   patBoolean noDerivative ,
				   const vector<patBoolean>& compBetaDerivatives,
				   const vector<patBoolean>& compParamDerivatives,
				   patBoolean compMuDerivative,
				   const vector<patBoolean>& compScaleDerivatives,
				   patVariables* betaDerivatives,
				   patVariables* paramDerivatives,
				   patReal* muDerivative,
				   patSecondDerivatives* secondDeriv,
				   patVariables* scaleDerivatives,
				   patBoolean* success,
				   patVariables* grad,
				   patBoolean computeBHHH,
				   vector<patVariables>* bhhh,
				   trHessian* trueHessian,
				   patError*& err) { // function call
  
  //Evaluate the loglikelihood function for given values of beta parameters
  //(involved in the utilities), model parameters (like mu_m parameters) and
  //scale parameters

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  if (!noDerivative && compMuDerivative) {
    if (muDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      return patReal()  ;
    }
  }

  if (!noDerivative) {
    if (betaDerivatives == NULL ||
	paramDerivatives == NULL ||
	scaleDerivatives == NULL ) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      return patReal()  ;
    }
  }

  if (betaParameters == NULL || 
      modelParameters == NULL || 
      scaleParameters == NULL ) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (mu == NULL) {
    err = new patErrNullPointer("patReal") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  

  if (sample == NULL) {
    err = new patErrNullPointer("patSample") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (model == NULL && panelModel == NULL) {
    err = new patErrNullPointer("patProbaModel") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (betaParameters->size() != patModelSpec::the()->getNbrTotalBeta()) {
    stringstream  str ;
    str << "There are " << betaParameters->size() 
	<< " instead of " << patModelSpec::the()->getNbrTotalBeta()  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }



  logLike = 0.0 ;
  nAlt = patModelSpec::the()->getNbrAlternatives() ;

  if (model != NULL) { //if (model != NULL) 

    if (!areObservationsAggregate) { //if (!areObservationsAggregate)
      observationCounter = 0 ;

      if (theObsIterator == NULL) {
	theObsIterator = sample->createObsIterator() ;
      }
      if (theObsIterator == NULL) {
	err = new patErrNullPointer(" patIterator<patObservationData*>*") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      step = patParameters::the()->getgevDataFileDisplayStep() ;

      //      ofstream debugObs("obsdebug.log") ;

      for (theObsIterator->first() ;
	   !theObsIterator->isDone() ;
	   theObsIterator->next()) { // Loop on all observations in the sample
	
	++observationCounter ;
	
	observation = theObsIterator->currentItem()  ;
	if (observation == NULL) {
	  err = new patErrNullPointer("patObservationData") ;
	  WARNING(err->describe()) ;
	  return patReal();
	}
	compScale = !noDerivative ;
	patReal* scaleDerivPointer = NULL ;
	groupIndex = sample->groupIndex(observation->group) ;
	if (groupIndex >= scaleParameters->size()) {
	  stringstream str ;
	  str << "Attempts to access group index " << groupIndex << " out of " 
	      << compScaleDerivatives.size()  ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING("*** Error for observation " << observationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	scale = (*scaleParameters)[groupIndex] ;
//  	DEBUG_MESSAGE("Obs: " << *observation) ;
//  	DEBUG_MESSAGE("Group: user " << observation->group << " index = " << groupIndex << " scale = " << scale) ;
	
	theScale = 
	  patModelSpec::the()->getScaleFromInternalId(groupIndex,err) ;
	if (err != NULL) {
	  WARNING("*** Error for observation " << observationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	scaleIndex = patBadId ;
	if (! theScale.isFixed) {
	  scaleIndex = theScale.index ; 
	}
	if (!noDerivative) {
	  compScale = compScaleDerivatives[groupIndex] ;
	  if (compScale) {
	    scaleDerivPointer = &((*scaleDerivatives)[groupIndex]) ;
	  }
	  
	}
	
	patValueVariables::the()->setAttributes(&(observation->attributes)) ;

	tmp = computeObservationProbability( observation,
					     NULL,
					     betaParameters,
					     mu,
					     modelParameters,
					     scaleParameters,
					     noDerivative ,
					     compBetaDerivatives,
					     compParamDerivatives,
					     compMuDerivative,
					     compScaleDerivatives,
					     betaDerivatives,
					     paramDerivatives,
					     muDerivative,
					     scaleDerivPointer,
					     secondDeriv,
					     success,
					     grad,
					     computeBHHH,
					     bhhh,
					     err) ;

	logLike += tmp ;

	//	debugObs << tmp << endl ;

	if (!noDerivative) {


	  patModelSpec::the()->gatherGradient(grad,
					      computeBHHH,
					      bhhh,
					      trueHessian,
					      betaDerivatives,
					      scaleDerivPointer,
					      scaleIndex,
					      paramDerivatives,
					      muDerivative,
					      secondDeriv,
					      err) ;

	}
	
	
	if (err != NULL) {
	  WARNING("*** Error for observation " << observationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal()  ;
	}
	
      }

//       debugObs.close() ;
//       DEBUG_MESSAGE("***** EXIT FOR DEBUGGING **********") ;
      
      //    DELETE_PTR(theObsIterator) ;
      
      if (!patFinite(logLike)) {
	logLike = -patMaxReal ;
      }
      
      if (!patFinite(logLike)) {
	stringstream str ;
	str << "Error in loglikelihood: " << logLike ;
	err = new patErrMiscError(str.str()) ;
	(*success) = patFALSE ;
	WARNING("*** Error for observation " << observationCounter) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      (*success) = patTRUE ;
      
      patDisplay::the().terminateProgressReport() ;
      return logLike  ;
    }
    else {
      // Aggregate observations
      aggObservationCounter = 0 ;
      observationCounter = 0 ;
      
      if (theAggObsIterator == NULL) {
	err = new patErrNullPointer(" patIterator<patAggObservationData*>*") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      step = patParameters::the()->getgevDataFileDisplayStep() ;
      
      //      DEBUG_MESSAGE("+++++ LOOP ON AGGREGATE OBSERVATIONS +++++") ;

      for (theAggObsIterator->first() ;
	   !theAggObsIterator->isDone() ;
	   theAggObsIterator->next()) { // Loop on all observations in the sample
	
	++aggObservationCounter ;

	aggObservation = theAggObsIterator->currentItem()  ;
	if (aggObservation == NULL) {
	  err = new patErrNullPointer("patAggregateObservationData") ;
	  WARNING(err->describe()) ;
	  return patReal();
	}
	observationCounter += aggObservation->getSize() ;

	compScale = !noDerivative ;
	patReal* scaleDerivPointer = NULL ;
	groupIndex = sample->groupIndex(aggObservation->getGroup()) ;
	if (groupIndex >= scaleParameters->size()) {
	  stringstream str ;
	  str << "Attempts to access group index " << groupIndex << " out of " 
	      << compScaleDerivatives.size()  ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING("*** Error for aggreg. observation " << aggObservationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	scale = (*scaleParameters)[groupIndex] ;
	
	theScale = 
	  patModelSpec::the()->getScaleFromInternalId(groupIndex,err) ;
	if (err != NULL) {
	  WARNING("*** Error for aggreg. observation " << aggObservationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	scaleIndex = patBadId ;
	if (! theScale.isFixed) {
	  scaleIndex = theScale.index ; 
	}
	if (!noDerivative) {
	  compScale = compScaleDerivatives[groupIndex] ;
	  if (compScale) {
	    scaleDerivPointer = &((*scaleDerivatives)[groupIndex]) ;
	  }
	  
	}
	
	
	tmp = computeObservationProbability( NULL,
					     aggObservation,
					     betaParameters,
					     mu,
					     modelParameters,
					     scaleParameters,
					     noDerivative ,
					     compBetaDerivatives,
					     compParamDerivatives,
					     compMuDerivative,
					     compScaleDerivatives,
					     betaDerivatives,
					     paramDerivatives,
					     muDerivative,
					     scaleDerivPointer,
					     secondDeriv,
					     success,
					     grad,
					     computeBHHH,
					     bhhh,
					     err) ;

	
// 	DEBUG_MESSAGE(observationCounter << " BEFORE: logLike=" << logLike) ;
// 	DEBUG_MESSAGE(observationCounter << " BEFORE: tmp=" << tmp) ;
	logLike += tmp ;
// 	DEBUG_MESSAGE(observationCounter << " AFTER: " << logLike) ;
	
	if (!noDerivative) {
	  patModelSpec::the()->gatherGradient(grad,
					      computeBHHH,
					      bhhh,
					      trueHessian,
					      betaDerivatives,
					      scaleDerivPointer,
					      scaleIndex,
					      paramDerivatives,
					      muDerivative,
					      secondDeriv,
					      err) ;

	}
	

	if (err != NULL) {
	  WARNING("*** Error for observation " << observationCounter) ;
	  WARNING(err->describe()) ;
	  return patReal()  ;
	}
	
      }
      
      //    DELETE_PTR(theObsIterator) ;
      
      if (!patFinite(logLike)) {
	logLike = -patMaxReal ;
      }
      
      if (!isfinite(logLike)) {
	stringstream str ;
	str << "Error in loglikelihood: " << logLike ;
	err = new patErrMiscError(str.str()) ;
	(*success) = patFALSE ;
	WARNING("*** Error for observation " << observationCounter) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      (*success) = patTRUE ;
      
      patDisplay::the().terminateProgressReport() ;
      return logLike  ;
    }
  }
  else if (panelModel != NULL) {

     // Loop on all observations in the sample
    individualCounter = 0 ;
    
    if (theIndIterator == NULL) {
      theIndIterator = sample->createIndIterator() ;
    }
    if (theIndIterator == NULL) {
      err = new patErrNullPointer(" patIterator<patIndividualData*>*") ;
    }
    for (theIndIterator->first() ;
 	 !theIndIterator->isDone() ;
 	 theIndIterator->next()) {
      
      
       ++individualCounter ;
      
       patIndividualData* observation = theIndIterator->currentItem()  ;
      
       compScale = !noDerivative ;
       patReal* scaleDerivPointer = NULL ;
       groupIndex = sample->groupIndex(observation->getGroup()) ;
       if (groupIndex >= scaleParameters->size()) {
 	stringstream str ;
 	str << "Attempts to access group index " << groupIndex << " out of " 
 	    << compScaleDerivatives.size()  ;
 	err = new patErrMiscError(str.str()) ;
 	WARNING("*** Error for individual " << individualCounter) ;
 	WARNING(err->describe()) ;
 	return patReal() ;
       }
       scale = (*scaleParameters)[groupIndex] ;
      
       

       if (!noDerivative) {
 	compScale = compScaleDerivatives[groupIndex] ;
 	if (compScale) {
 	  scaleDerivPointer = &((*scaleDerivatives)[groupIndex]) ;
 	}
	
 	theScale = 
 	  patModelSpec::the()->getScaleFromInternalId(groupIndex,err) ;
 	if (err != NULL) {
 	  WARNING("*** Error for individual " << individualCounter) ;
 	  WARNING(err->describe()) ;
 	  return patReal() ;
 	}
 	scaleIndex = patBadId ;
 	if (! theScale.isFixed) {
 	  scaleIndex = theScale.index ; 
 	}
       }
      
       tmp = computePanelIndividualProbability( observation,
						betaParameters,
						mu,
						modelParameters,
						scaleParameters,
						noDerivative ,
						compBetaDerivatives,
						compParamDerivatives,
						compMuDerivative,
						compScaleDerivatives,
						betaDerivatives,
						paramDerivatives,
						muDerivative,
						scaleDerivPointer,
						success,
						grad,
						computeBHHH,
						bhhh,
						err) ;

       if (err != NULL) {
	 WARNING(err->describe()) ;
	 return patReal();
       }

       logLike += tmp ;


       if (!noDerivative) {
	 patModelSpec::the()->gatherGradient(grad,
					     computeBHHH,
					     bhhh,
					     NULL,
					     betaDerivatives,
					     scaleDerivPointer,
					     scaleIndex,
					     paramDerivatives,
					     muDerivative,
					     NULL,
					     err) ;
	 if (err != NULL) {
	   WARNING("*** Error for individual " << individualCounter) ;
	   WARNING(err->describe()) ;
	   return patReal()  ;
	 }
       }
      
      
     }
    
    //     DELETE_PTR(theIndIterator) ;
    
    if (!isfinite(logLike)) {
      logLike = -patMaxReal ;
    }
    
    (*success) = patTRUE ;
    patDisplay::the().terminateProgressReport() ;
    return logLike  ;
    
  }

  WARNING("This statement should not be reached") ;
  WARNING(err->describe()) ;
  return patReal()  ;

}
  
patString patLikelihood::getModelName(patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return 0;
  }
  
  if (model != NULL) {
    return model->getModelName(err) ;
  }
  else if (panelModel != NULL) {
    return panelModel->getModelName(err) ;
  }
  else {
    err = new patErrNullPointer("patProbaModel") ;
    WARNING(err->describe()) ;
    return 0;
  }
  

}


patReal patLikelihood::getTrivialModelLikelihood() {
  // Loop on all observations in the sample
  patReal logLike(0.0) ;
  if (areObservationsAggregate) {
    assert(theAggObsIterator != NULL) ;
    for (theAggObsIterator->first() ;
	 !theAggObsIterator->isDone() ;
	 theAggObsIterator->next()) {
      patAggregateObservationData* aggObservation = theAggObsIterator->currentItem() ;
      patReal oneProba(0.0) ;
      for (vector<patObservationData>::iterator i = 
	     aggObservation->theObservations.begin() ;
	   i != aggObservation->theObservations.end() ;
	   ++i) {
	patReal choiceSetSize(0.0) ;
	if (patModelSpec::the()->isOL()) {
	  choiceSetSize = 
	    patModelSpec::the()->getOrdinalLogitNumberOfIntervals() ;
	}
	else {
	  for (vector<patBoolean>::iterator j = i->availability.begin() ;
	       j != i->availability.end() ;
	       ++j) {
	    if (*j) {
	      choiceSetSize += 1.0 ;
	    }
	  }
	}
	oneProba += i->aggWeight / choiceSetSize ;
      }
      logLike += aggObservation->getWeight() * log(oneProba) ;
    }
  }
  else {
    if (theObsIterator == NULL) {
      theObsIterator = sample->createObsIterator() ;
    }
    assert(theObsIterator != NULL) ;
    for (theObsIterator->first() ;
	 !theObsIterator->isDone() ;
	 theObsIterator->next()) {
      patObservationData* observation = theObsIterator->currentItem() ;
      patReal choiceSetSize(0.0) ;
      if (patModelSpec::the()->isOL()) {
	choiceSetSize = 
	  patModelSpec::the()->getOrdinalLogitNumberOfIntervals() ;
      }
      else {
	for (vector<patBoolean>::iterator i = observation->availability.begin() ;
	     i != observation->availability.end() ;
	     ++i) {
	  if (*i) {
	    choiceSetSize += 1.0 ;
	  }
	}
      }
      logLike += observation->weight * log(1.0 / choiceSetSize) ;
    }
    //DELETE_PTR(theObsIterator) ;
  }
  return logLike ;
}




patReal patLikelihood::computeObservationProbability(patObservationData* observation,
						     patAggregateObservationData* aggObservation,
						     patVariables* betaParameters,
						     const patReal* mu,
						     const patVariables* modelParameters,
						     const patVariables* scaleParameters,
						     patBoolean noDerivative ,
						     const vector<patBoolean>& compBetaDerivatives,
						     const vector<patBoolean>& compParamDerivatives,
						     patBoolean compMuDerivative,
						     const vector<patBoolean>& compScaleDerivatives,
						     patVariables* betaDerivatives,
						     patVariables* paramDerivatives,
						     patReal* muDerivative,
						     patReal* scaleDerivPointer,
						     patSecondDerivatives* secondDeriv,
						     patBoolean* success,
						     patVariables* grad,
						     patBoolean computeBHHH,
						     vector<patVariables>* bhhh,
						     patError*& err) {
  
  if (!noDerivative) {
    fill(betaDerivatives->begin(),betaDerivatives->end(),0.0) ;
    fill(paramDerivatives->begin(),paramDerivatives->end(),0.0) ;
    if (compMuDerivative) {
      *muDerivative = 0.0 ;
    } 
    if (compScale) {
      *scaleDerivPointer = 0.0 ;
    }
  }
  
  tmp = 0.0  ;
  
  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
    
    if (theDiscreteParamModel == NULL) {
      err = new patErrNullPointer("patDiscreteParameteProba") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    tmp = theDiscreteParamModel->evalProbaLog(observation,
					      aggObservation,
					      betaParameters,
					      modelParameters,
					      scale,
					      scaleIndex,
					      noDerivative,
					      compBetaDerivatives,
					      compParamDerivatives,
					      compMuDerivative,
					      compScale,
					      betaDerivatives,
					      paramDerivatives,
					      muDerivative,
					      scaleDerivPointer,
					      mu,
					      secondDeriv,
					      success,
					      grad,
					      computeBHHH,
					      bhhh,
					      err) ;

//     int debug = 0 ;
//     for (theCombination->first() ;
// 	 !theCombination->isDone() ;
// 	 theCombination->next()) {  // loop on combinations
//       ++debug ;
//       oneCombi =  theCombination->currentItem() ;
      
//       weightProbabilityOfTheCombination = 1.0 ;
      
//       for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
// 	   i < oneCombi->size() ;
// 	   ++i) {
	
// 	patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
	
// 	patString name = patModelSpec::the()->getDiscreteParamName(i,err) ;
// 	if (err != NULL) {
// 	  WARNING("*** Error for observation " << observationCounter) ;
// 	  WARNING(err->describe()) ;
// 	DEBUG_MESSAGE("Return 1 ") ;
// 	  return patReal();
// 	}
	
// 	patModelSpec::the()->setDiscreteParameterValue(name,
// 						       theTerm->massPoint,
// 						       err) ;
// 	if (err != NULL) {
// 	  WARNING("*** Error for observation " << observationCounter) ;
// 	  WARNING(err->describe()) ;
// 	DEBUG_MESSAGE("Return 2 ") ;
// 	  return patReal() ;
// 	}
// 	weightProbabilityOfTheCombination *= theTerm->probability->estimated ;
//       }
      
//       if (weightProbabilityOfTheCombination > 0) { // if nonzero proba
	
// 	patReal oneLogProba = model->evalProbaLog(observation,
// 						  aggObservation,
// 						  betaParameters,
// 						  modelParameters,
// 						  scale,
// 						  scaleIndex,
// 						  noDerivative,
// 						  compBetaDerivatives,
// 						  compParamDerivatives,
// 						  compMuDerivative,
// 						  compScale,
// 						  &singleBetaDerivatives,
// 						  &singleParamDerivatives,
// 						  &singleMuDerivative,
// 						  &singleScaleDeriv,
// 						  mu,
// 						  secondDeriv,
// 						  success,
// 						  grad,
// 						  computeBHHH,
// 						  bhhh,
// 						  err) ;
	    
// 	DEBUG_MESSAGE("oneLogProba = " << oneLogProba) ;

// 	if (err != NULL) {
// 	  WARNING("*** Error for observation " << observationCounter) ;
// 	  WARNING(err->describe()) ;
// 	DEBUG_MESSAGE("Return 3 ") ;
// 	  return patReal() ;
// 	}
	    
// 	if (!(*success)) {
// 	DEBUG_MESSAGE("Return 4 ") ;
// 	  return patMaxReal ;
// 	}
	    

// 	patReal oneProba = exp(oneLogProba) ;
// 	patReal correctedProba = weightProbabilityOfTheCombination * oneProba ;

// 	tmp += correctedProba ;
	    

// 	if (!noDerivative) { // noderivative
	      

// 	  // Affect the derivative with respect to the discretely
// 	  // distributed parameter to the appropriate parameters
	      
// 	  for (vector<vector<patDiscreteTerm>::iterator>::size_type i = 0 ;
// 	       i < oneCombi->size() ;
// 	       ++i) {
// 	    patDiscreteTerm* theTerm = &*(*oneCombi)[i] ;
// 	    unsigned long actualBetaId = theTerm->massPoint->id ;
// 	    patReal theDiscreteProba = theTerm->probability->estimated ;
// 	    unsigned long discreteId = discreteParamId[i] ;
		
// 	    singleBetaDerivatives[actualBetaId] = singleBetaDerivatives[discreteId] ;
	       
// 	    if (compBetaDerivatives[theTerm->probability->id]) {
// 	      singleBetaDerivatives[theTerm->probability->id] = 1 / theTerm->probability->estimated ;
// 	    }
// 	  }
	      
// 	  //Multiply all derivatives by the probability
// 	  // Note that the derivatives are computed in the function for log(P). Therefore, they also have to be multiplied by P.
	      
	      
// 	  for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
// 	    if (compBetaDerivatives[i]) {
// 	      singleBetaDerivatives[i] *= correctedProba ;
// 	    }
// 	  }
// 	  for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
// 	    if (compParamDerivatives[i]) {
// 	      singleParamDerivatives[i] *= correctedProba ;
// 	    }
// 	  }
// 	  if (compMuDerivative) {
// 	    singleMuDerivative *= correctedProba ;
// 	  }
// 	  if (compScale) {
// 	    singleScaleDeriv *= correctedProba ;
// 	  }
	      
// 	  // Accumulate the derivatives
	      
	      
// 	  for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
// 	    if (compBetaDerivatives[i]) {
// 	      (*betaDerivatives)[i] += singleBetaDerivatives[i] ;
// 	      singleBetaDerivatives[i] = 0.0 ;
// 	    }
// 	  }
// 	  for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
// 	    if (compParamDerivatives[i]) {
// 	      (*paramDerivatives)[i] += singleParamDerivatives[i] ;
// 	      singleParamDerivatives[i] = 0.0 ;
// 	    }
// 	  }
// 	  if (compMuDerivative) {
// 	    (*muDerivative) += singleMuDerivative ;
// 	    singleMuDerivative = 0.0 ;
// 	  }
// 	  if (compScale) {
// 	    (*scaleDerivPointer) += singleScaleDeriv ;
// 	    singleScaleDeriv = 0.0 ;
// 	  }
// 	}
//       }
//     }


//     if (!noDerivative) {
//       for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
// 	if (compBetaDerivatives[i]) {
// 	  (*betaDerivatives)[i] /= tmp ;
// 	}
//       }
//       for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
// 	if (compParamDerivatives[i]) {
// 	  (*paramDerivatives)[i] /= tmp ;
// 	}
//       }
//       if (compMuDerivative) {
// 	(*muDerivative) /= tmp ;
//       }
//       if (compScale) {
// 	(*scaleDerivPointer) /= tmp ;
//       }
//     }
	
//     tmp = log(tmp) ;

  }      
  else { // else
    tmp = model->evalProbaLog(observation,
			      aggObservation,
			      betaParameters,
			      modelParameters,
			      scale,
			      scaleIndex,
			      noDerivative,
			      compBetaDerivatives,
			      compParamDerivatives,
			      compMuDerivative,
			      compScale,
			      betaDerivatives,
			      paramDerivatives,
			      muDerivative,
			      scaleDerivPointer,
			      mu,
			      secondDeriv,
			      success,
			      grad,
			      computeBHHH,
			      bhhh,
			      err) ;

    if (err != NULL) {
      WARNING("*** Error for observation " << observationCounter) ;
      WARNING(err->describe()) ;
      DEBUG_MESSAGE("Return 5 ") ;
      return patReal() ;
    }
  }

      
  if (err != NULL) {
    WARNING("*** Error for observation " << observationCounter) ;
    WARNING(err->describe()) ;
	DEBUG_MESSAGE("Return 5 ") ;
    return patReal() ;
  }
      
  return tmp ;
}


/////////////////////////////////

patReal patLikelihood::computePanelIndividualProbability(patIndividualData* observation,
							 patVariables* betaParameters,
							 const patReal* mu,
							 const patVariables* modelParameters,
							 const patVariables* scaleParameters,
							 patBoolean noDerivative ,
							 const vector<patBoolean>& compBetaDerivatives,
							 const vector<patBoolean>& compParamDerivatives,
							 patBoolean compMuDerivative,
							 const vector<patBoolean>& compScaleDerivatives,
							 patVariables* betaDerivatives,
							 patVariables* paramDerivatives,
							 patReal* muDerivative,
							 patReal* scaleDerivPointer,
							 patBoolean* success,
							 patVariables* grad,
							 patBoolean computeBHHH,
							 vector<patVariables>* bhhh,
							 patError*& err) {


  if (!noDerivative) {
    fill(betaDerivatives->begin(),betaDerivatives->end(),0.0) ;
    fill(paramDerivatives->begin(),paramDerivatives->end(),0.0) ;
    if (compMuDerivative) {
      *muDerivative = 0.0 ;
    }
    if (compScale) {
      *scaleDerivPointer = 0.0 ;
    }
  }
  
  tmp = 0.0  ;
  
  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
    
    tmp = theDiscreteParamModel->evalPanelProba(observation,
						betaParameters,
						modelParameters,
						scale,
						scaleIndex,
						noDerivative,
						compBetaDerivatives,
						compParamDerivatives,
						compMuDerivative,
						compScale,
						betaDerivatives,
						paramDerivatives,
						muDerivative,
						scaleDerivPointer,
						mu,
						success,
						grad,
						computeBHHH,
						bhhh,
						err) ;    

    if (err != NULL) {
      WARNING("*** Error for observation " << observationCounter) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
    if (!(*success)) {
      return patMaxReal ;
    }
    
    

//     if (!noDerivative) {
//       for (unsigned long i = 0 ; i < compBetaDerivatives.size() ; ++i) {
// 	if (compBetaDerivatives[i]) {
// 	  (*betaDerivatives)[i] /= tmp ;
// 	}
//       }
//       for (unsigned long i = 0 ; i < compParamDerivatives.size() ; ++i) {
// 	if (compParamDerivatives[i]) {
// 	  (*paramDerivatives)[i] /= tmp ;
// 	}
//       }
//       if (compMuDerivative) {
// 	(*muDerivative) /= tmp ;
//       }
//       if (compScale) {
// 	(*scaleDerivPointer) /= tmp ;
//       }
      
//     }
    
  }
  else { // else
    tmp = panelModel->evalProbaLog(observation,
				   betaParameters,
				   modelParameters,
				   scale,
				   scaleIndex,
				   noDerivative,
				   compBetaDerivatives,
				   compParamDerivatives,
				   compMuDerivative,
				   compScale,
				   betaDerivatives,
				   paramDerivatives,
				   muDerivative,
				   scaleDerivPointer,
				   mu,
				   success,
				   grad,
				   computeBHHH,
				   bhhh,
				   err) ;


    if (err != NULL) {
      WARNING("*** Error for observation " << observationCounter) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
  }
  
  
  
  if (!(*success)) {
    return patMaxReal ;
  }
  return tmp ;
}


void patLikelihood::generateCppCode(ostream& cppFile, 
				    patError*& err) {

  patBoolean trueHessian = (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian() && patModelSpec::the()->isMuFixed()) ;

  cppFile << "//////////////////////////////////" << endl ;
  cppFile << "// Code generated in patLikelihood" << endl ;


  nAlt = patModelSpec::the()->getNbrAlternatives() ;
  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;
  patBoolean isPanel = patModelSpec::the()->isPanelData() ;

  if (model != NULL || panelModel != NULL) { //if (model != NULL) 

    if (!areObservationsAggregate) { //if (!areObservationsAggregate)

      cppFile << "void *computeFunction(void* fctPtr) {" << endl ;
      cppFile << "" << endl ;
      if (!patModelSpec::the()->groupScalesAreAllOne()) {
	cppFile << "  vector<unsigned long> groupIndex("<<patModelSpec::the()->getLargestGroupUserId() + 1<<") ;" << endl ;
	cppFile << "  vector<patReal> scalesPerGroup("<<patModelSpec::the()->getLargestGroupUserId() + 1<<") ;" << endl ;
	patModelSpec::the()->generateCppCodeForGroupId(cppFile,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      cppFile << "  vector<unsigned long> altIndex("<<patModelSpec::the()->getLargestAlternativeUserId() + 1 << ") ;" << endl ;
      patModelSpec::the()->generateCppCodeForAltId(cppFile,err) ;
      cppFile << "  inputArg *input;" << endl ;
      cppFile << "  input = (inputArg *) fctPtr;" << endl ;
      cppFile << "" << endl ;
      cppFile << "  trVector* x =  input->x ;" << endl ;
      cppFile << "  trVector* grad = input->grad ;" << endl ;
      cppFile << "  patBoolean* success = input->success ;" << endl ;
      if (trueHessian) {
	cppFile << "  trHessian* trueHessian = input->trueHessian ;" << endl ;
      }
      cppFile << "  trHessian* bhhh = input->bhhh ;" << endl ;
      cppFile << "  patError*& err = input->err ;" << endl ;
      cppFile << "" << endl ;
      if (!isPanel) {
	cppFile << "  patObservationData* observation ;" << endl ;
      }
      else {
	cppFile << "  patIndividualData* individual ;" << endl ;

      }


      cppFile << "  patReal logLike = 0.0 ;" << endl ;
      cppFile << "" << endl ;
      if (!isPanel) {
	cppFile << "  for (input->theObsIterator->first() ;" << endl ;
	cppFile << "       !input->theObsIterator->isDone() ;" << endl ;
	cppFile << "       input->theObsIterator->next()) { // Loop on all observations in the sample" << endl ;
      cppFile << "    " << endl ;
      cppFile << "    observation = input->theObsIterator->currentItem()  ;" << endl ;
      cppFile << "    if (observation == NULL) {" << endl ;
      cppFile << "      input->err = new patErrNullPointer(\"patObservationData\") ;" << endl ;
      cppFile << "      WARNING(err->describe()) ;" << endl ;
      cppFile << "      return 0 ;" << endl ;
      cppFile << "    }" << endl ;
      cppFile << "" << endl ;
      }
      else {
	cppFile << "  for (input->theIndIterator->first() ;" << endl ;
	cppFile << "       !input->theIndIterator->isDone() ;" << endl ;
	cppFile << "       input->theIndIterator->next()) { // Loop on all individuals in the sample" << endl ;
	cppFile << "    " << endl ;
	cppFile << "    individual = input->theIndIterator->currentItem()  ;" << endl ;
	cppFile << "    if (individual == NULL) {" << endl ;
	cppFile << "      input->err = new patErrNullPointer(\"patIndividualData\") ;" << endl ;
	cppFile << "      WARNING(err->describe()) ;" << endl ;
	cppFile << "      return 0 ;" << endl ;
	cppFile << "    }" << endl ;
	cppFile << "" << endl ;
      }
      cppFile << "    patReal logProbaOneObs ;" << endl ;
	
      generateCppCodeOneObservation(cppFile, patFALSE, patFALSE, err) ;
      
      
      cppFile << "      logLike += logProbaOneObs ;" << endl ;
	
      
      
      cppFile << "      if (err != NULL) {" << endl ;
      cppFile << "	WARNING(err->describe()) ;" << endl ;
      cppFile << "	return NULL  ;" << endl ;
      cppFile << "      }" << endl ;
      
      cppFile << "    if (!patFinite(logLike) || finite(logLike) == 0) {" << endl ;
      cppFile << "      logLike = -patMaxReal ;" << endl ;
      cppFile << "    }" << endl ;
      cppFile << "      " << endl ;
      cppFile << "      (*success) = patTRUE ;" << endl ;
      cppFile << "      " << endl ;
      cppFile << "    }" << endl ;
      cppFile << "  input->result = logLike ;" << endl ;
      cppFile << "    }" << endl ;


      cppFile << "void *computeFunctionAndGradient(void* fctPtr) {" << endl ;
      cppFile << "" << endl ;
      if (!patModelSpec::the()->groupScalesAreAllOne()) {
	cppFile << "  vector<unsigned long> groupIndex("<<patModelSpec::the()->getLargestGroupUserId() + 1<<") ;" << endl ;
	cppFile << "  vector<patReal> scalesPerGroup("<<patModelSpec::the()->getLargestGroupUserId() + 1<<") ;" << endl ;
	patModelSpec::the()->generateCppCodeForGroupId(cppFile,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      cppFile << "  vector<unsigned long> altIndex("<<patModelSpec::the()->getLargestAlternativeUserId() + 1 << ") ;" << endl ;
      patModelSpec::the()->generateCppCodeForAltId(cppFile,err) ;
      cppFile << "  inputArg *input;" << endl ;
      cppFile << "  input = (inputArg *) fctPtr;" << endl ;
      cppFile << "" << endl ;
      cppFile << "  trVector* x =  input->x ;" << endl ;
      cppFile << "  trVector* grad = input->grad ;" << endl ;
      if (trueHessian) {
	cppFile << "  trHessian* trueHessian = input->trueHessian ;" << endl ;
      }
      cppFile << "  trHessian* bhhh = input->bhhh ;" << endl ;
      cppFile << "  patBoolean* success = input->success ;" << endl ;
      cppFile << "  patError*& err = input->err ;" << endl ;
      cppFile << "" << endl ;
      if (!isPanel) {
	cppFile << "  patObservationData* observation ;" << endl ;
      }
      else {
	cppFile << "  patIndividualData* individual ;" << endl ;

      }


      cppFile << "  patReal logLike = 0.0 ;" << endl ;
      cppFile << "" << endl ;
      if (!isPanel) {
	cppFile << "  for (input->theObsIterator->first() ;" << endl ;
	cppFile << "       !input->theObsIterator->isDone() ;" << endl ;
	cppFile << "       input->theObsIterator->next()) { // Loop on all observations in the sample" << endl ;
	cppFile << "    " << endl ;
	cppFile << "    observation = input->theObsIterator->currentItem()  ;" << endl ;
	cppFile << "    if (observation == NULL) {" << endl ;
	cppFile << "      input->err = new patErrNullPointer(\"patObservationData\") ;" << endl ;
	cppFile << "      WARNING(err->describe()) ;" << endl ;
	cppFile << "      return 0 ;" << endl ;
	cppFile << "    }" << endl ;
	cppFile << "" << endl ;

      }
      else {
	cppFile << "  for (input->theIndIterator->first() ;" << endl ;
	cppFile << "       !input->theIndIterator->isDone() ;" << endl ;
	cppFile << "       input->theIndIterator->next()) { // Loop on all observations in the sample" << endl ;
	cppFile << "    " << endl ;
	cppFile << "    individual = input->theIndIterator->currentItem()  ;" << endl ;
	cppFile << "    if (individual == NULL) {" << endl ;
	cppFile << "      input->err = new patErrNullPointer(\"patIndividualData\") ;" << endl ;
	cppFile << "      WARNING(err->describe()) ;" << endl ;
	cppFile << "      return 0 ;" << endl ;
	cppFile << "    }" << endl ;
	cppFile << "" << endl ;

      }
      cppFile << "    patReal logProbaOneObs ;" << endl ;
      cppFile << "    trVector gradientLogOneObs(" << K << ",0.0) ;" << endl ;
	
      generateCppCodeOneObservation(cppFile, patTRUE, patTRUE, err) ;
      
      cppFile << "    logLike += logProbaOneObs ;" << endl ;
      cppFile << "	for (unsigned long k = 0 ; k < "<< K <<"  ; ++k) {" << endl ;
      cppFile << "	  for (unsigned long kk = k ; kk < "<< K <<" ; ++kk) {" << endl ;
      cppFile << "	    bhhh->addElement(k,kk,gradientLogOneObs[k] * gradientLogOneObs[kk],err) ;" << endl ;
      cppFile << "	  }" << endl ;
      cppFile << "	} // for (unsigned long k = 0 ; k < "<< K <<" ; ++k)" << endl ;
      cppFile << "	(*grad) += gradientLogOneObs ;" << endl ;
      cppFile << "      if (err != NULL) {" << endl ;
      cppFile << "	WARNING(err->describe()) ;" << endl ;
      cppFile << "	return 0  ;" << endl ;
      cppFile << "      }" << endl ;
      cppFile << "    if (isinf(logLike) == -1 || finite(logLike) == 0) {" << endl ;
      cppFile << "      logLike = -patMaxReal ;" << endl ;
      cppFile << "    }" << endl ;
      cppFile << "      " << endl ;
      cppFile << "    (*success) = patTRUE ;" << endl ;
      cppFile << "      " << endl ;
      cppFile << "  }" << endl ;
      cppFile << "  input->result = logLike ;" << endl ;
      cppFile << "      " << endl ;
      cppFile << "    }" << endl ;
	
      
      
      cppFile << "//////////////////////////////////" << endl ;
      cppFile << "// End of code generated in patLikelihood" << endl ;
      cppFile << "//////////////////////////////////" << endl ;


    }  //endif (!areObservationsAggregate)
    

    else {
      err = new patErrMiscError("Not yet implemented") ;
      WARNING(err->describe());
      return ;

    }
  }
}

void patLikelihood::generateCppCodeOneObservation(ostream& cppFile,
						  patBoolean derivatives, 
						  patBoolean secondDerivatives, 
						  patError*& err) {


  // Objective: write the code to compute probaOneObs and, if necessary,
  // gradientLogOneObs
  // Simply calls the function implemented in the model

  if (patModelSpec::the()->containsDiscreteParameters()) { // contains discrete paramters
    cppFile << "    // Discrete parameters" << endl ;
    if (theDiscreteParamModel == NULL) {
      err = new patErrNullPointer("patDiscreteParameteProba") ;
      WARNING(err->describe()) ;
      return  ;
    }
    theDiscreteParamModel->generateCppCode(cppFile,derivatives,err) ;
    
  }      
  else { // else
    if (model != NULL) {
      model->generateCppCode(cppFile,derivatives,secondDerivatives,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
    }
    else if (panelModel != NULL) {
      panelModel->generateCppCode(cppFile,derivatives,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
    }
    else {
      err = new patErrMiscError("No model defined") ;
      WARNING(err->describe()) ;
      return ;
    }
  }

  cppFile << "// End of code generated in patLikelihood" << endl ;
  cppFile << "//////////////////////////////////" << endl ;

}

// void patLikelihood::generateScaleCppCode(ostream& cppFile,
// 					 patError*& err) {

//   for (list<long>::iterator i = patModelSpec::the()->groupIds.begin() ;
//        i != patModelSpec::the()->groupIds.end() ;
//        ++i) {
//     unsigned long groupIndex = sample->groupIndex(*i) ;
//     patBetaLikeParameter scale = 
//       patModelSpec::the()->getScaleFromInternalId(groupIndex,
// 						  err)  ;
//     if (err != NULL) {
//       WARNING(err->describe()) ;
//       return ;
//     }
//     cppFile << "// ****** HERE ***** //" << endl ;
//     cppFile << "      groupIndex["<<*i<<"] = "<<scale.index<<";" << endl ;
//     cppFile << "      scalesPerGroup["<<*i<<"] = "<<scale.defaultValue<<";" << endl ;
    
//   }
// }

