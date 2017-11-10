//-*-c++-*------------------------------------------------------------
//
// File name : patProbaProbitModel.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 23 00:08:01 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patProbaProbitModel.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"
#include "patUtility.h"

patProbaProbitModel::patProbaProbitModel(patUtility* aUtility) :
  patProbaModel(aUtility), V(2) {

}

patProbaProbitModel::~patProbaProbitModel() {

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
    
patReal patProbaProbitModel::evalProbaPerDraw(patBoolean logOfProba, 
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
    err = new patErrMiscError("SNP not yet implemented for Probit") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (patModelSpec::the()->getNbrAlternatives() != 2) {
    err = new patErrMiscError("Probit can be estimated for binary models only.") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  
  static patBoolean first = patTRUE ;
  if (first) {
    nBeta = beta->size() ;
    utilDerivativesChosen.resize(nBeta) ;
    utilDerivativesUnchosen.resize(nBeta) ;
    first = patFALSE  ;
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
  
  chosen = patModelSpec::the()->getAltInternalId(index,err) ;
  unchosen = (chosen == 1) ? 0 : 1 ;
  
  if (!individual->availability[chosen]) {
    err = new patErrMiscError("Chosen alternative not available") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!individual->availability[unchosen]) {
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
  
  
  diff = (V[chosen] - V[unchosen]) / *mu ;
  
  proba = patNormalCdf::the()->compute(diff,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (!noDerivative) {
    fill(utilDerivativesChosen.begin(),utilDerivativesChosen.end(),0.0) ;
    utility->computeBetaDerivative(indivId,
				   drawNumber,
				   chosen,
				   beta,
				   x,
				   &utilDerivativesChosen,
				   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    fill(utilDerivativesUnchosen.begin(),utilDerivativesUnchosen.end(),0.0) ;
    utility->computeBetaDerivative(indivId,
				   drawNumber,
				   unchosen,
				   beta,
				   x,
				   &utilDerivativesUnchosen,
				   err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    
    for ( k = 0 ; k < nBeta ; ++k) {
      if (compBetaDerivatives[k]) {
	if (logOfProba) {
	  (*betaDerivatives)[k] += 
	    (patNormalCdf::the()->derivative(diff,err) * 
	     (utilDerivativesChosen[k] - utilDerivativesUnchosen[k])) / (*mu * proba) ;
	}
	else {
	  (*betaDerivatives)[k] +=
	    (patNormalCdf::the()->derivative(diff,err) * 
	     (utilDerivativesChosen[k] - utilDerivativesUnchosen[k])) / *mu ;
	}
      }
      if (compMuDerivative) {
	err = new patErrMiscError("Not yet implemented") ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
    }
  }
  if (logOfProba) {
    return log(proba) ;
  }
  return proba ;
}





patString patProbaProbitModel::getModelName(patError*& err) {
  return patString("Binary probit model") ;
}

patString  patProbaProbitModel::getInfo() {
  return patString("No specific information available") ;
}

void patProbaProbitModel::generateCppCodePerDraw(ostream& str,
						 patBoolean logOfProba,
						 patBoolean derivatives, 
						 patBoolean secondDerivatives, 
						 patError*& err) {
  err = new patErrMiscError("Feature not implemented for probit models") ;
  WARNING(err->describe()) ;
  return ;
}
