//-*-c++-*------------------------------------------------------------
//
// File name : patProbaOrdinalLogit.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 23 00:08:01 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patProbaOrdinalLogit.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"
#include "patUtility.h"
#include "patBetaLikeParameter.h"

patProbaOrdinalLogit::patProbaOrdinalLogit(patUtility* aUtility) :
  patProbaModel(aUtility), V(2) {

}

patProbaOrdinalLogit::~patProbaOrdinalLogit() {
  
}


/**
   Evaluates the logarithm of the probability given by the model that the
   individiual chooses alternative 'index', knowing the utilities, for a given draw number. If
   requested, the derivatives are evaluated as well. Note that the derivatives
   are cumulated to previous values. In order to have the value of the
   derivatives, the corresponding storage area must be initialized to zero
   before calling this method.
   
   @param logOfProba if patTRUE, the log of the probability is returned
   @param indivId identifier of the individual
   @param drawNumber number of the requested draw
   @param index index of the chosen alternative
   @param utilities value of the utility of each alternative
   @param beta value of the beta parameters
   @param x value of the characteristics
   @param parameters value of the structural parameters of the GEV function
   @param scale scale parameter
   @param noDerivative patTRUE: no derivative is computed. 
   patFALSE: derivatives may be computed.
   @param compBetaDerivatives If entry k is patTRUE, the derivative with 
   respect to $\beta_k$ is computed
   @param compParamDerivatives If entry k is patTRUE, the derivative with 
   respect to parameter $k$ is computed
   @param compMuDerivative If patTRUE, the derivative with respect to $\mu$ 
   is computed.
   @param betaDerivatives pointer to the vector where the derivatives with respoect to $\beta$ will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and for entries corresponding to patTRUE in compBetaDerivatives. All other cells are undefined. 
   @param Derivatives pointer to the vector where the derivatives with respect to the GEV parameters will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and for entries corresponding to patTRUE in compParamDerivatives. All other cells are undefined. 
   @param muDerivative pointer to the patReal where the derivative with respect to mu will be accumulated. The result will be meaningful only if noDerivatives is patFALSE and  compMuDerivative is patTRUE.
   @param available describes which alternative are available
   @param mu value of the homogeneity parameter of the GEV function
   @param err ref. of the pointer to the error object. 
   @return pointer to the variable where the result will be stored. 
   Same as outputvariable
   
   
   
*/
    
patReal patProbaOrdinalLogit::evalProbaPerDraw(patBoolean logOfProba, 
					      patObservationData* individual,
					      unsigned long drawNumber,
					      patVariables* beta,
					      const patVariables* parameters,
					      patReal scale,
					      patBoolean noDerivative ,
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
					      patBoolean snpTerms,
					      patReal factorForDerivOfSnpTerms,
					      patBoolean* success,
					      patError*& err) {
  

  if (snpTerms) {
    err = new patErrMiscError("SNP not yet implemented for Ordinal Logit") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (patModelSpec::the()->getNbrAlternatives() != 2) {
    err = new patErrMiscError("Ordinal logit can be estimated for binary models only.") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  
  static patBoolean first = patTRUE ;
  if (first) {
    nBeta = beta->size() ;
    utilDerivativesZero.resize(nBeta) ;
    utilDerivativesOne.resize(nBeta) ;
    first = patFALSE  ;

    GENERAL_MESSAGE("Ordinal logit with " << patModelSpec::the()->getOrdinalLogitNumberOfIntervals() << " intervals") ;

    // Build the list of interval, with each time the pameter
    // corresponding ot the lower and upper bound

    map<unsigned long, patBetaLikeParameter*>* ordLogitThresholds = 
      patModelSpec::the()->getOrdinalLogitThresholds() ;
    unsigned long nextOne = patModelSpec::the()->getOrdinalLogitLeftAlternative() ;
    pair<patBetaLikeParameter*,patBetaLikeParameter*> thePair(NULL,NULL) ;
    for (map<unsigned long, patBetaLikeParameter*>::iterator i =
	   ordLogitThresholds->begin() ;
	 i != ordLogitThresholds->end() ;
	 ++i) {
      thePair.second = i->second ;
      thresholds[nextOne] = thePair ;
      thePair.first = thePair.second ;
      nextOne = i->first ;
    }
    thePair.second = NULL ;
    thresholds[nextOne] = thePair ;
    DEBUG_MESSAGE("Intervals for ordinal logit") ;
    for (map<unsigned long, 
    pair<patBetaLikeParameter*,patBetaLikeParameter*> >::iterator i = 
	   thresholds.begin() ;
	 i != thresholds.end() ;
	 ++i) {
      patString left = (i->second.first == NULL)
	? "$NONE"
	: i->second.first->name ;
	patString right = (i->second.second == NULL)
	  ?"$NONE"
	  : i->second.second->name ;
      DEBUG_MESSAGE(left << "  --> " << right) ;
		    
    }
  }
  
  
  x = &individual->attributes ;
  index = individual->choice ;
  weight = individual->weight ;
  indivId = individual->id ;
  
  if (utility == NULL) {
    err = new patErrNullPointer("patUtility") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  

  if (!(individual->availability[0] && individual->availability[1]) ) {
    if (logOfProba) {
      return 0.0 ;
    }
    else {
      return 1.0 ;
    }
  }
  
  
  V[0] = utility->computeFunction(indivId,
				  drawNumber,
				  0,
				  beta,
				  x,
				  err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  V[1] = utility->computeFunction(indivId,
				  drawNumber,
				  1,
				  beta,
				  x,
				  err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  

  diff = V[0] - V[1] ;


  map<unsigned long, 
    pair<patBetaLikeParameter*,patBetaLikeParameter*> >::iterator found = 
    thresholds.find(index) ;
  if (found == thresholds.end()) {
    stringstream str ;
    str << "Interval " << index << " not defined for the Ordinal Logit" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
//   patString left = (found->second.first == NULL)
//     ? "$NONE"
//     : found->second.first->name ;
//   patString right = (found->second.second == NULL)
//     ?"$NONE"
//     : found->second.second->name ;
//   DEBUG_MESSAGE("Index " << index << ": " << left << "  --> " << right) ;
  patReal xx = 0 ;
  patReal yy = 0; 
  patReal denomx ;
  patReal denomy ;
  if (found->second.first != NULL) {
    xx = -(*mu) * ( diff - found->second.first->estimated ) ;
    if (xx >= patLogMaxReal::the()) {
      denomx = patMaxReal ;
      proba = 0.0 ;
    }
    else {
      denomx = 1.0 + exp(xx) ;
      proba = 1.0 / denomx ;
    }
  }
  else {
    proba = 1.0 ;
  }
  if (found->second.second != NULL) {
    yy = -(*mu) * ( diff - found->second.second->estimated ) ;
    if (yy >= patLogMaxReal::the()) {
      denomy = patMaxReal ;
    }
    else {
      denomy = 1.0 + exp(yy) ;
      proba -= 1.0 / denomy ;
    }
  }
  
  //  DEBUG_MESSAGE("proba=" << proba) ;
  if (!noDerivative) { // !noDerivative
    unsigned long xid = (found->second.first == NULL)
      ? patBadId
      : found->second.first->id ;
    unsigned long yid = (found->second.second == NULL)
      ? patBadId
      : found->second.second->id ;

    fill(utilDerivativesZero.begin(),utilDerivativesZero.end(),0.0) ;
    utility->computeBetaDerivative(indivId,
				   drawNumber,
				   patULong(0),
				   beta,
				   x,
				   &utilDerivativesZero,
				   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    fill(utilDerivativesOne.begin(),utilDerivativesOne.end(),0.0) ;
    utility->computeBetaDerivative(indivId,
				   drawNumber,
				   patULong(1),
				   beta,
				   x,
				   &utilDerivativesOne,
				   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
    for ( k = 0 ; k < nBeta ; ++k) {
      if (compBetaDerivatives[k]) { //if (compBetaDerivatives[k])
	if (logOfProba) {
	  if (found->second.first != NULL) {	    
	    if (xx < patLogMaxReal::the()) {
	      (*betaDerivatives)[k] -= exp(xx) * (*mu) * 
		(utilDerivativesOne[k] - utilDerivativesZero[k]) 
		/ (denomx * denomx * proba) ;
	    }
	  }
	  if (found->second.second != NULL) {
	    
	    if (yy < patLogMaxReal::the()) {
	      (*betaDerivatives)[k] += exp(yy) * (*mu) * 
		(utilDerivativesOne[k] - utilDerivativesZero[k]) 
		/ (denomy * denomy * proba) ;
	    }
	  }
	}
	else {
	  if (found->second.first != NULL) {
	    if (xx < patLogMaxReal::the()) {
	      (*betaDerivatives)[k] -= exp(xx) * (*mu) * 
		(utilDerivativesOne[k] - utilDerivativesZero[k]) 
		/ (denomx * denomx) ;
	    }
	  }
	  if (found->second.second != NULL) {
	    if (yy < patLogMaxReal::the()) {
	      (*betaDerivatives)[k] += exp(yy) * (*mu) * 
		(utilDerivativesOne[k] - utilDerivativesZero[k]) 
		/ (denomy * denomy) ;
	    }
	  }
	}
      }
    }
      
    // Derivatives with respect to the threshold parameters
    if (found->second.first != NULL) {
      if (compBetaDerivatives[xid]) {
	if (logOfProba) {
	  if (xx < patLogMaxReal::the()) {
	    (*betaDerivatives)[xid] -= exp(xx) * (*mu) / (denomx * denomx * proba) ;
	  }
	}
	else {
	  if (xx < patLogMaxReal::the()) {
	    (*betaDerivatives)[xid] -= exp(xx) * (*mu) / (denomx * denomx) ;
	  }
	}
	
      }
    }
    if (found->second.second != NULL) {
      if (compBetaDerivatives[yid]) {
	if (logOfProba) {
	  if (yy < patLogMaxReal::the()) {
	    (*betaDerivatives)[yid] += exp(yy) * (*mu) / (denomy * denomy * proba) ;
	  }
	}
	else {
	  if (yy < patLogMaxReal::the()) {
	    (*betaDerivatives)[yid] += exp(yy) * (*mu) / (denomy * denomy) ;
	  }
	}
      }
    }
    if (compMuDerivative) {
      err = new patErrMiscError("Not yet implemented") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  if (logOfProba) {
    return log(proba) ;
  }
  return proba ;
}





patString patProbaOrdinalLogit::getModelName(patError*& err) {
  return patString("Ordinal logit model") ;
}

patString  patProbaOrdinalLogit::getInfo() {
  return patString() ;
}

void patProbaOrdinalLogit::generateCppCodePerDraw(ostream& str,
						  patBoolean logOfProba,
						  patBoolean derivatives, 
						  patBoolean secondDerivatives, 
						  patError*& err) {
  err = new patErrMiscError("Feature not implemented for ordinal logit models") ;
  WARNING(err->describe()) ;
  return ;
}
