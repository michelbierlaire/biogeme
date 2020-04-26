//-*-c++-*------------------------------------------------------------
//
// File name : patProbaPanelModel.cc
// Author :    Michel Bierlaire
// Date :      Tue Mar 30 16:24:14 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patProbaPanelModel.h"
#include "patDisplay.h"
#include "patModelSpec.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patRandomDraws.h"
#include "patVariables.h"


patProbaPanelModel::patProbaPanelModel(patUtility* aUtility) : utility(aUtility),betaDrawDerivatives(NULL), paramDrawDerivatives(NULL),muDrawDerivative(NULL),scaleDrawDerivative(NULL),betaSingleDerivatives(NULL), paramSingleDerivatives(NULL),muSingleDerivative(NULL),scaleSingleDerivative(NULL),betaAggregDerivatives(NULL), paramAggregDerivatives(NULL),muAggregDerivative(NULL),scaleAggregDerivative(NULL) {
  idOfSnpBetaParameters = patModelSpec::the()->getIdOfSnpBetaParameters() ;
  derivNormalizationConstant.resize(idOfSnpBetaParameters.size()) ;

  useAggregateObservation = patModelSpec::the()->isAggregateObserved() ;
}

patProbaPanelModel::~patProbaPanelModel() {
  DELETE_PTR(betaDrawDerivatives) ;
  DELETE_PTR(paramDrawDerivatives) ;
  DELETE_PTR(muDrawDerivative) ;
  DELETE_PTR(scaleDrawDerivative) ;
  DELETE_PTR(betaSingleDerivatives) ;
  DELETE_PTR(paramSingleDerivatives) ;
  DELETE_PTR(muSingleDerivative) ;
  DELETE_PTR(scaleSingleDerivative) ;
  DELETE_PTR(betaAggregDerivatives) ;
  DELETE_PTR(paramAggregDerivatives) ;
  DELETE_PTR(muAggregDerivative) ;
  DELETE_PTR(scaleAggregDerivative) ;

}


void patProbaPanelModel::setUtility(patUtility* aUtility) {
  utility = aUtility ;
}

patUtility* patProbaPanelModel::getUtility() {
  return utility ;
}


patString patProbaPanelModel::getInfo() {
  return patString("No information available") ;
}


patReal patProbaPanelModel::evalProbaLog(patIndividualData* individual,
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


  if (individual == NULL) {
    err = new patErrNullPointer("patObservationData") ;
  }

  patReal weight = individual->getWeight() ;

  unsigned long R = (patModelSpec::the()->isMixedLogit()) 
    ? patModelSpec::the()->getAlgoNumberOfDraws() 
    : 1 ;
  

  if (!noDerivative) {
    if (betaDerivatives != NULL) {
      if ( betaDrawDerivatives == NULL) {
	betaDrawDerivatives = new patVariables(betaDerivatives->size(),0.0) ;
      }
      else {
	fill(betaDrawDerivatives->begin(),betaDrawDerivatives->end(),0.0) ;
      }
    }
    if (paramDerivatives != NULL) {
      if(paramDrawDerivatives == NULL) {
	paramDrawDerivatives = new patVariables(paramDerivatives->size(),0.0) ;
      }
      else {
	fill(paramDrawDerivatives->begin(),paramDrawDerivatives->end(),0.0) ;
      }
    }
    if (muDrawDerivative == NULL) {
      muDrawDerivative = new patReal(0.0) ;
    }
    else {
      *muDrawDerivative = 0.0 ;
    }
    if (scaleDrawDerivative == NULL) {
      scaleDrawDerivative = new patReal(0.0) ;
    }
    else {
      *scaleDrawDerivative = 0.0 ;
    }
  }

  if (patModelSpec::the()->applySnpTransform()) {
    normalizeSnpTerms = 1.0 ;
    for (unsigned short term = 0 ;
	 term < idOfSnpBetaParameters.size() ;
	 ++term) {
      normalizeSnpTerms += (*beta)[idOfSnpBetaParameters[term]] * (*beta)[idOfSnpBetaParameters[term]] ;
    }
    for (unsigned short term = 0 ;
	 term < idOfSnpBetaParameters.size() ;
	 ++term) {
      derivNormalizationConstant[term] = -2.0 * (*beta)[idOfSnpBetaParameters[term]] / normalizeSnpTerms ;
    }
  }

  patReal sumOfProba(0.0) ;

  for (unsigned long drawNumber = 1 ; drawNumber <= R ; ++drawNumber) {
    
    
    //    DEBUG_MESSAGE("Ind " << individual->id << " draw " << drawNumber) ;
    
    if (patModelSpec::the()->applySnpTransform()) {
      qForSnp = 1.0 ;
      for (unsigned short term = 0 ;
	   term < idOfSnpBetaParameters.size() ;
	   ++term) {
	qForSnp += (*beta)[idOfSnpBetaParameters[term]] * 
	  //It is assumed that the tested parameter is distributed
	  //across individuals, and not across observations
	  individual->getUnifDrawsForSnpPolynomial(drawNumber,term) ;

      }
      snpCorrection = qForSnp * qForSnp / normalizeSnpTerms ;
    }
    
     proba = evalProbaPerDraw(individual,
			      drawNumber,
			      beta,
			      parameters,
			      scale,
			      noDerivative ,
			      compBetaDerivatives,
			      compParamDerivatives,
			      compMuDerivative,
			      compScaleDerivative,
			      betaDrawDerivatives,
			      paramDrawDerivatives,
			      muDrawDerivative,
			      scaleDrawDerivative,
			      mu,
			      patModelSpec::the()->applySnpTransform(),
			      snpCorrection,
			      success,
			      err)  ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }

      if (!isfinite(proba)) {
	DEBUG_MESSAGE("proba = " << proba) ;
      }

    //    DEBUG_MESSAGE("proba per draw = " << tmp) ;

    if (patModelSpec::the()->applySnpTransform()) {
      tmp = proba * snpCorrection ;
      for (unsigned short term = 0 ;
	   term < idOfSnpBetaParameters.size() ;
	   ++term) {
	(*betaDrawDerivatives)[idOfSnpBetaParameters[term]] +=
	  2 * proba * qForSnp * individual->getUnifDrawsForSnpPolynomial(drawNumber,term) / normalizeSnpTerms ;
      }      
    }
    else {
      tmp = proba ;
    }

    sumOfProba += tmp ;


  }

  // DERIVATIVES

  if (!noDerivative) {
    if (betaDerivatives != NULL) {
      if (patModelSpec::the()->applySnpTransform()) {
	for (unsigned short term = 0 ;
	     term < idOfSnpBetaParameters.size() ;
	     ++term) {
	  (*betaDrawDerivatives)[idOfSnpBetaParameters[term]] += -2.0 * (*beta)[idOfSnpBetaParameters[term]] * sumOfProba / normalizeSnpTerms ;
	  }	  
      }	      
      (*betaDerivatives) += (weight / sumOfProba) * (*betaDrawDerivatives)  ;
      for (unsigned short i = 0 ;
	   i < betaDerivatives->size() ;
	   ++i) {
	if (!isfinite((*betaDerivatives)[i])) {
	  DEBUG_MESSAGE("weight=" << weight) ;
	  DEBUG_MESSAGE("sumOfProba=" << sumOfProba) ;
	  DEBUG_MESSAGE("betaDrawDerivatives[" << i << "]=" <<(*betaDrawDerivatives)[i]) ;
	}
      }
    }
    if (paramDerivatives != NULL) {
      (*paramDerivatives) += (weight / sumOfProba) * (*paramDrawDerivatives)  ;
    }
    if (compMuDerivative) {
      (*muDerivative) += weight * (*muDrawDerivative) / sumOfProba ;
    }
    if (compScaleDerivative) {
      (*scaleDerivative) += weight * (*scaleDrawDerivative) / sumOfProba ;
    }

//     patModelSpec::the()->gatherGradient(grad,
// 					computeBHHH,
// 					bhhh,
// 					betaDerivatives,
// 					scaleDerivative,
// 					scaleIndex,
// 					paramDerivatives,
// 					muDerivative,
// 					err) ;
  }


  
  return (weight * log(sumOfProba / patReal(R))) ;

  
  
}




patReal patProbaPanelModel::evalProbaPerDraw( patIndividualData* individual,
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
					      patBoolean snpTerms,
					      patReal factorForDerivOfSnpTerms,
					      patBoolean* success,
					      patError*& err) {
  
  /**
     We denote $f_i$ the probability associated with observation
     $i$ of the current individual, and by $g$ the probability
     associated with the individual.
     
     \[
     g = \prod_i f_i = e^{\sum_i \ln f_i}
     \]
     and 
     \[
     \frac{\partial g}{\partial x} = g \sum_i \frac{1}{f_i}
     \frac{\partial f_i}{\partial x}
     \]
  */



   patReal sumLnFi(0.0) ;
   
   patVariables logg_betaDerivatives ;
   if (betaDerivatives != NULL) {
     logg_betaDerivatives.resize(betaDerivatives->size(),0.0) ;
   }
   patVariables logg_paramDerivitaves ;
   if (paramDerivatives != NULL) {
     logg_paramDerivitaves.resize(paramDerivatives->size(),0.0) ;
   }
   patReal logg_muDerivative(0.0) ;
   patReal logg_scaleDerivative(0.0) ;
   
   if (useAggregateObservation) { // useAggregateObservation
    for (vector<patAggregateObservationData>::iterator i = 
	   individual->theAggregateObservations.begin() ;
	 i != individual->theAggregateObservations.end() ;
	 ++i) {
      if (!noDerivative) {
	f_betaDerivatives.resize(betaDerivatives->size()) ;
	fill(f_betaDerivatives.begin(),
	     f_betaDerivatives.end(),
	     0.0) ;
	f_paramDerivatives.resize(paramDerivatives->size()) ;
	fill(f_paramDerivatives.begin(),
	     f_paramDerivatives.end(),
	     0.0) ;
      }
      patReal f_muDerivative(0.0) ;
      patReal f_scaleDerivative(0.0) ;
      
      patReal fi =  evalProbaPerAggObs(NULL,
				       &(*i),
				       drawNumber,
				       beta,
				       parameters,
				       scale,
				       noDerivative ,
				       compBetaDerivatives,
				       compParamDerivatives,
				       compMuDerivative,
				       compScaleDerivative,
				       &f_betaDerivatives,
				       &f_paramDerivatives,
				       &f_muDerivative,
				       &f_scaleDerivative,
				       mu,
				       success,
				       err) ;

      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }

      if (!isfinite(fi) || !isfinite(log(fi))) {
	DEBUG_MESSAGE("fi = " << fi) ;
      }
      
      sumLnFi += log(fi) ;
      if (!noDerivative) {
	for (unsigned long i = 0 ; i < betaDerivatives->size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    add = f_betaDerivatives[i] / fi ;
	    if (isfinite(add)) {
	      logg_betaDerivatives[i] += add;
	      if (!isfinite(logg_betaDerivatives[i] )) {
		DEBUG_MESSAGE("f_betaDerivatives["<<i<<"]=" << f_betaDerivatives[i]) ;
		DEBUG_MESSAGE("fi = " << fi) ;
	      }
	    }
	  }
	}
	for (unsigned long i = 0 ; i < paramDerivatives->size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    add = f_paramDerivatives[i] / fi ;
	    if (isfinite(add)) {
	      logg_paramDerivitaves[i] += add ;
	    }
	  }
	}
	if (compMuDerivative) {
	  add = f_muDerivative / fi ;
	  if (isfinite(add)) {
	    logg_muDerivative += add ;
	  }
	}
	if (compScaleDerivative) {
	  add = f_scaleDerivative / fi ;
	  if (isfinite(add)) {
	    logg_scaleDerivative += add ;
	  }
	}
      }
    }
  }
  else {
    
    for (vector<patObservationData>::iterator i = 
	   individual->theObservations.begin() ;
	 i != individual->theObservations.end() ;
	 ++i) {
      if (!noDerivative) {
	f_betaDerivatives.resize(betaDerivatives->size()) ;
	fill(f_betaDerivatives.begin(),
	     f_betaDerivatives.end() ,
	     0.0) ;
	f_paramDerivatives.resize(paramDerivatives->size()) ;
	fill(f_paramDerivatives.begin(),
	     f_paramDerivatives.end() ,
	     0.0) ;
      }
      patReal f_muDerivative(0.0) ;
      patReal f_scaleDerivative(0.0) ;
      patReal fi =  evalProbaPerAggObs(&(*i),
				       NULL,
				       drawNumber,
				       beta,
				       parameters,
				       scale,
				       noDerivative ,
				       compBetaDerivatives,
				       compParamDerivatives,
				       compMuDerivative,
				       compScaleDerivative,
				       &f_betaDerivatives,
				       &f_paramDerivatives,
				       &f_muDerivative,
				       &f_scaleDerivative,
				       mu,
				       success,
				       err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      if (!isfinite(fi) || !isfinite(log(fi))) {
	DEBUG_MESSAGE("fi = " << fi) ;
      }

      sumLnFi += log(fi) ;
      if (!noDerivative) {
	for (unsigned long i = 0 ; i < betaDerivatives->size() ; ++i) {
	  if (compBetaDerivatives[i]) {
	    add = f_betaDerivatives[i] / fi ;
	    if (isfinite(add)) {
	      logg_betaDerivatives[i] += add;
	      if (!isfinite(logg_betaDerivatives[i] )) {
		DEBUG_MESSAGE("f_betaDerivatives["<<i<<"]=" << f_betaDerivatives[i]) ;
		DEBUG_MESSAGE("fi = " << fi) ;
	      }
	    }
	  }
	}
	for (unsigned long i = 0 ; i < paramDerivatives->size() ; ++i) {
	  if (compParamDerivatives[i]) {
	    add = f_paramDerivatives[i] / fi ;
	    if (isfinite(add)) {
	      logg_paramDerivitaves[i] += add ;
	    }
	  }
	}
	if (compMuDerivative) {
	  add = f_muDerivative / fi ;
	  if (isfinite(add)) {
	    logg_muDerivative += add ;
	  }
	}
	if (compScaleDerivative) {
	  add = f_scaleDerivative / fi ;
	  if (isfinite(add)) {
	    logg_scaleDerivative += add ;
	  }
	}
	//      DEBUG_MESSAGE("logg_betaDerivatives = " << logg_betaDerivatives) ;
      }
    }
  }

  //  DEBUG_MESSAGE("sumLnFi= " <<sumLnFi) ;

  patReal g = exp(sumLnFi) ; 
  //  DEBUG_MESSAGE("g=" << g) ;
  if (!noDerivative) {
    for (unsigned long i = 0 ; i < betaDerivatives->size() ; ++i) {
      if (compBetaDerivatives[i]) {
	if (snpTerms) {
	  (*betaDerivatives)[i] += logg_betaDerivatives[i] * g * factorForDerivOfSnpTerms;
	}
	else {
	  (*betaDerivatives)[i] += logg_betaDerivatives[i] * g ;
	  if (!isfinite((*betaDerivatives)[i])) {
	    DEBUG_MESSAGE("logg_betaDerivatives["<<i<<"]=" << logg_betaDerivatives[i]) ;
	    DEBUG_MESSAGE("g=" <<g) ;
	  }
	}
      }
    }
    for (unsigned long i = 0 ; i < paramDerivatives->size() ; ++i) {
      if (compParamDerivatives[i]) {
	if (snpTerms) {
	  (*paramDerivatives)[i] += logg_paramDerivitaves[i] * g * factorForDerivOfSnpTerms;
	}
	else {
	  (*paramDerivatives)[i] += logg_paramDerivitaves[i] * g ;
	}
      }
    }
    if (compMuDerivative) {
      if (snpTerms) {
	(*muDerivative) += logg_muDerivative * g * factorForDerivOfSnpTerms;
      }
      else{
	(*muDerivative) += logg_muDerivative * g ;
      }
    }
    if (compScaleDerivative) {
      if (snpTerms) {
	(*scaleDerivative) += logg_scaleDerivative * g * factorForDerivOfSnpTerms;
      }
      else {
	(*scaleDerivative) += logg_scaleDerivative * g ;
      }
    }

    //    DEBUG_MESSAGE("betaDerivatives= " <<*betaDerivatives) ;
  }

  //  DEBUG_MESSAGE("exp(sum)=" << g) ;
  
  return g ;
}



patReal patProbaPanelModel::evalProbaPerAggObs(patObservationData* observation,
					       patAggregateObservationData* aggObservation,
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
					       patBoolean* success,
					       patError*& err) {

  if (observation != NULL) {
    patReal result =  evalProbaPerObs(observation,
				      drawNumber,
				      beta,
				      parameters,
				      scale,
				      noDerivative ,
				      compBetaDerivatives,
				      compParamDerivatives,
				      compMuDerivative,
				      compScaleDerivative,
				      betaDerivatives,
				      paramDerivatives,
				      muDerivative,
				      scaleDerivative,
				      mu,
				      success,
				      err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
      if (!isfinite(result)) {
	DEBUG_MESSAGE("result = " << result) ;
      }
    return result ;
  }

  // Aggregate observation

  if (aggObservation == NULL) {
    err = new patErrNullPointer("patObservationData") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  else {
    // Compute proba for an aggregate observation
    patReal proba(0.0) ;
    if (!noDerivative) { // if (!noDerivative)
      if (betaDerivatives != NULL) {
	if (betaAggregDerivatives == NULL) {
	  betaAggregDerivatives = new patVariables(betaDerivatives->size(),0.0) ;
	}
	else {
	  fill(betaAggregDerivatives->begin(),betaAggregDerivatives->end(),0.0) ;
	}
      }
      if (paramDerivatives != NULL) {
	if (paramAggregDerivatives == NULL) {
	  paramAggregDerivatives = new patVariables(paramDerivatives->size(),0.0) ;
	}
	else {
	  fill(paramAggregDerivatives->begin(),paramAggregDerivatives->end(),0.0) ;
	}
      }
      if (muAggregDerivative == NULL) {
	muAggregDerivative = new patReal(0.0) ;
      }
      else {
	*muAggregDerivative  = 0.0 ;
      }
      if (scaleAggregDerivative == NULL) {
	scaleAggregDerivative = new patReal(0.0) ;
      }
      else {
	*scaleAggregDerivative = 0.0 ;
      }
    }
    
    for (vector<patObservationData>::iterator i = 
	   aggObservation->theObservations.begin() ;
	 i != aggObservation->theObservations.end() ;
	 ++i) { // loop on aggreg obs
      if (!noDerivative) { // if (!noDerivative)
	if (betaDerivatives != NULL) {
	  if (betaSingleDerivatives == NULL) {
	    betaSingleDerivatives = new patVariables(betaDerivatives->size(),0.0) ;
	  }
	  else {
	    fill(betaSingleDerivatives->begin(),betaSingleDerivatives->end(),0.0) ;
	  }
	}
	if (paramDerivatives != NULL) {
	  if (paramSingleDerivatives == NULL) {
	    paramSingleDerivatives = new patVariables(paramDerivatives->size(),0.0) ;
	  }
	  else {
	    fill(paramSingleDerivatives->begin(),paramSingleDerivatives->end(),0.0) ;
	  }
	}
	if (muSingleDerivative == NULL) {
	  muSingleDerivative = new patReal(0.0) ;
	}
	else {
	  *muSingleDerivative  = 0.0 ;
	}
	if (scaleSingleDerivative == NULL) {
	  scaleSingleDerivative = new patReal(0.0) ;
	}
	else {
	  *scaleSingleDerivative = 0.0 ;
	}
      }
      patReal oneProba = evalProbaPerObs(&(*i),
					    drawNumber,
					    beta,
					    parameters,
					    scale,
					    noDerivative ,
					    compBetaDerivatives,
					    compParamDerivatives,
					    compMuDerivative,
					    compScaleDerivative,
					    betaSingleDerivatives,
					    paramSingleDerivatives,
					    muSingleDerivative,
					    scaleSingleDerivative,
					    mu,
					    success,
					    err) ;

      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      if (!isfinite(oneProba)) {
	DEBUG_MESSAGE("oneProba = " << oneProba) ;
      }
      //      DEBUG_MESSAGE("proba= " << oneProba) ;
      //       DEBUG_MESSAGE("weight = " << i->aggWeight) ;
      patReal weightedOneProba = i->aggWeight * oneProba ;
      //       DEBUG_MESSAGE("probaBefore = " << proba) ;
      proba += weightedOneProba ;
      //       DEBUG_MESSAGE("probaAfter = " << proba) ;
      
      if (!noDerivative) {
	(*betaAggregDerivatives) += i->aggWeight * (*betaSingleDerivatives) ;
	(*paramAggregDerivatives) += i->aggWeight * (*paramSingleDerivatives) ;
	*muAggregDerivative += i->aggWeight * (*muSingleDerivative) ;
	*scaleAggregDerivative += i->aggWeight * (*scaleSingleDerivative) ;
      }
    }
    
    if (!noDerivative) {
      if (betaDerivatives != NULL) {
	(*betaDerivatives) += (*betaAggregDerivatives) ;
      }
      if (paramDerivatives != NULL) {
	(*paramDerivatives) += (*paramAggregDerivatives)  ;
      }
      if (muDerivative != NULL) {
	(*muDerivative) += (*muAggregDerivative) ;
      }
      if (scaleDerivative != NULL) {
	(*scaleDerivative) += (*scaleAggregDerivative)  ;
      }
    }
    return proba ;
  }
}

void patProbaPanelModel::generateCppCode(ostream& cppFile,
					 patBoolean derivatives, 
					 patError*& err) {
  
  // Objective: write the code to compute logProbaOneObs and, if necessary,
  // gradientLogOneObs

  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;

  cppFile << "///////////////////////////////////////" << endl ;
  cppFile << "// Code generated in patProbaPanelModel" << endl ;

  cppFile << "    patReal weight = individual->getWeight() ;" << endl ;
  
  unsigned long R = (patModelSpec::the()->isMixedLogit()) 
    ? patModelSpec::the()->getAlgoNumberOfDraws() 
    : 1 ;
  if (R <= 1) {
    stringstream str ;
    str << "The number of draws is " << R << ". It should be greater than 1" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }
  if (patModelSpec::the()->applySnpTransform()) {
    err = new patErrMiscError("Not yet implemented") ;
    WARNING(err->describe()) ;
    return ;
  }  

  cppFile << "  patReal sumOfProba(0.0) ;" << endl ;
  if (derivatives) {
    cppFile << "  patVariables sumOfGradient(" << K << ",0.0) ;" << endl ;
  }


  cppFile << "  for (unsigned long drawNumber = 1 ; drawNumber <= "
	  << R 
	  << "  ; ++drawNumber) {" 
	  << endl ;
  
  generateCppCodePerDraw(cppFile,derivatives, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  cppFile << "    sumOfProba += probaPerDraw ;" << endl ;
  if (derivatives) {
    cppFile << "    sumOfGradient += gradientPerDraw ;" << endl ;
  }
  cppFile << "    logProbaOneObs =  weight * (log(sumOfProba) - log(patReal(" << R << "))) ;" << endl ;
  if (derivatives) {
    cppFile << "gradientLogOneObs = (weight / sumOfProba) * sumOfGradient ;" ;
  }
  
  cppFile << "  }  " << endl ;

  cppFile << "// End of code generated in patProbaPanelModel" << endl ;
  cppFile << "//////////////////////////////////////////////" << endl ;
}

void patProbaPanelModel::generateCppCodePerDraw(ostream& cppFile,
						   patBoolean derivatives, 
						   patError*& err) {
  
  cppFile << "//////////////////////////////////////////////" << endl ;
  cppFile << "// Code generated in generateCppCodePerDraw " << endl ;
  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;
  cppFile << "  patReal sumOfLogs(0.0) ;" << endl ;
  if (derivatives) {
    cppFile << "  patVariables gradientPerDraw(" << K << ",0.0) ;" << endl ;
  }
  cppFile << "  for (vector<patObservationData>::iterator i = " << endl ;
  cppFile << "	 individual->theObservations.begin() ;" << endl ;
  cppFile << "       i != individual->theObservations.end() ;" << endl ;
  cppFile << "       ++i) {" << endl ;
  cppFile << "  patObservationData* observation = &(*i) ;" << endl ;
  generateCppCodePerObs(cppFile,derivatives,err) ;
  cppFile << "  }" << endl ;
  cppFile << "  patReal probaPerDraw = exp(sumOfLogs) ;" << endl ;
  if (derivatives) {
    cppFile << "  for (unsigned long k = 0 ; k < " << K << " ; ++k) {" << endl ;
    cppFile << "  gradientPerDraw[k] *= probaPerDraw ;" << endl ;
    cppFile << "} " << endl ;
  }
  cppFile << "// End of Code generated in generateCppCodePerDraw " << endl ;
  cppFile << "//////////////////////////////////////////////" << endl ;
}
