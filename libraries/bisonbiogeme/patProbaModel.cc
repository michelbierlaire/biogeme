//-*-c++-*------------------------------------------------------------
//
// File name : patProbaModel.cc
// Author :    Michel Bierlaire
// Date :      Mon Jun  4 16:36:24 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patProbaModel.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"
#include "patValueVariables.h"

patProbaModel::patProbaModel(patUtility* aUtility) : utility(aUtility),betaDrawDerivatives(NULL), paramDrawDerivatives(NULL),muDrawDerivative(NULL),scaleDrawDerivative(NULL),betaSingleDerivatives(NULL), paramSingleDerivatives(NULL),muSingleDerivative(NULL),scaleSingleDerivative(NULL),betaAggregDerivatives(NULL), paramAggregDerivatives(NULL),muAggregDerivative(NULL),scaleAggregDerivative(NULL) {
  idOfSnpBetaParameters = patModelSpec::the()->getIdOfSnpBetaParameters() ;
  derivNormalizationConstant.resize(idOfSnpBetaParameters.size()) ;
}

patProbaModel::~patProbaModel() {
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


void patProbaModel::setUtility(patUtility* aUtility) {
  utility = aUtility ;
}

patUtility* patProbaModel::getUtility() {
  return utility ;
}


patString patProbaModel::getInfo() {
  return patString("No information available") ;
}


patReal patProbaModel::evalProbaLog(patObservationData* observation,
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
  
  
  if (observation == NULL) {
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
	   ++i) {
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

      //      cout << "evalProbaLog scale= " << scale << " index = " << scaleIndex << endl ;

	patReal oneLogProba = evalProbaLog(&(*i),
					   NULL,
					   beta,
					   parameters,
					   scale,
					   scaleIndex,
					   noDerivative,
					   compBetaDerivatives,
					   compParamDerivatives,
					   compMuDerivative,
					   compScaleDerivative,
					   betaSingleDerivatives,
					   paramSingleDerivatives,
					   muSingleDerivative,
					   scaleSingleDerivative,
					   mu,
					   secondDeriv,
					   success,
					   grad,
					   computeBHHH,
					   bhhh,
					   err) ;

	//	DEBUG_MESSAGE("oneLogProba=" << oneLogProba) ;

	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	patReal oneProba = exp(oneLogProba) ;
	patReal weightedOneProba = i->aggWeight * oneProba ;
// 	DEBUG_MESSAGE("weight=" << i->aggWeight) ;
// 	DEBUG_MESSAGE("weightedOneProba=" << weightedOneProba) ;
	proba += weightedOneProba ;
//	DEBUG_MESSAGE("proba=" << proba) ;

	if (!noDerivative) {
	  (*betaAggregDerivatives) += weightedOneProba * (*betaSingleDerivatives) ;
	  (*paramAggregDerivatives) += weightedOneProba * (*paramSingleDerivatives) ;
	  *muAggregDerivative += weightedOneProba * (*muSingleDerivative) ;
	  //	  cout << "update " << *scaleAggregDerivative << " with " << *scaleSingleDerivative << endl ;
	  *scaleAggregDerivative += weightedOneProba * (*scaleSingleDerivative) ;
	}
      }

      if (!noDerivative) {
	if (betaDerivatives != NULL) {
	  (*betaDerivatives) += (*betaAggregDerivatives) / proba ;
	}
	if (paramDerivatives != NULL) {
	  (*paramDerivatives) += (*paramAggregDerivatives) / proba ;
	}
	if (muDerivative != NULL) {
	  (*muDerivative) += (*muAggregDerivative) / proba ;
	}
	if (scaleDerivative != NULL) {
	  //	  cout << "update " << *scaleDerivative << " with " << *scaleAggregDerivative << endl ;
	  (*scaleDerivative) += (*scaleAggregDerivative) / proba ;
	}
      }
      return log(proba) ;
    }
  }

  patValueVariables::the()->setAttributes(&(observation->attributes)) ;
  
  patReal weight = observation->weight ;

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
    patValueVariables::the()->setRandomDraws(&(observation->draws[drawNumber-1])) ;

    //    DEBUG_MESSAGE("Ind " << observation->id << " draw " << drawNumber) ;

    if (patModelSpec::the()->applySnpTransform()) {
      qForSnp = 1.0 ;
      for (unsigned short term = 0 ;
	   term < idOfSnpBetaParameters.size() ;
	   ++term) {
	qForSnp += (*beta)[idOfSnpBetaParameters[term]] * 
	  observation->unifDrawsForSnpPolynomial[drawNumber-1][term] ;
      }
      snpCorrection = qForSnp * qForSnp / normalizeSnpTerms ;
    }

    proba = evalProbaPerDraw((R == 1),
			     observation,
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
			     secondDeriv,
			     patModelSpec::the()->applySnpTransform(),
			     snpCorrection,
			     success,
			     err)  ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      DEBUG_MESSAGE("Return 1") ;
      return patReal() ;
    }

    if (patModelSpec::the()->applySnpTransform()) {
      tmp = proba * snpCorrection ;
      for (unsigned short term = 0 ;
	   term < idOfSnpBetaParameters.size() ;
	   ++term) {
	(*betaDrawDerivatives)[idOfSnpBetaParameters[term]] +=
	  2 * proba * qForSnp * observation->unifDrawsForSnpPolynomial[drawNumber-1][term] / normalizeSnpTerms ;
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
      if ( R > 1) {
	if (patModelSpec::the()->applySnpTransform()) {
	  for (unsigned short term = 0 ;
	       term < idOfSnpBetaParameters.size() ;
	       ++term) {
	    (*betaDrawDerivatives)[idOfSnpBetaParameters[term]] += -2.0 * (*beta)[idOfSnpBetaParameters[term]] * sumOfProba / normalizeSnpTerms ;
	  }	  
	}	
	(*betaDerivatives) += (weight / sumOfProba) * (*betaDrawDerivatives)  ;
      }
      else {
	(*betaDerivatives) += weight * (*betaDrawDerivatives)  ;
      }
    }
    if (paramDerivatives != NULL) {
      if (R > 1) {
	(*paramDerivatives) += (weight / sumOfProba) * (*paramDrawDerivatives)  ;
      }
      else {
	(*paramDerivatives) += weight * (*paramDrawDerivatives)  ;
      }
    }
    if (compMuDerivative) {
      if (R > 1) {
	(*muDerivative) += weight * (*muDrawDerivative) / sumOfProba ;
      }
      else {
	(*muDerivative) += weight * (*muDrawDerivative)  ;
      }
    }
    if (compScaleDerivative) {
      if (R > 1) {
	(*scaleDerivative) += weight * (*scaleDrawDerivative) / sumOfProba ;
      }
      else {
	(*scaleDerivative) += weight * (*scaleDrawDerivative) ;
      }
    }

  }


  if (R > 1) {
    return (weight * log(sumOfProba / patReal(R))) ;
  }
  else {
    return (weight * sumOfProba) ;
  }

  
  
}

void patProbaModel::generateCppCode(ostream& cppFile,
				    patBoolean derivatives, 
				    patBoolean secondDerivatives, 
				    patError*& err) {

  // Objective: write the code to compute logProbaOneObs and, if necessary,
  // gradientLogOneObs

  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;

  cppFile << "//////////////////////////////////" << endl ;
  cppFile << "// Code generated in patProbaModel" << endl ;
  
  if (patModelSpec::the()->isAggregateObserved()) {

    err = new patErrMiscError("Must implement for aggregate obs") ;
    WARNING(err->describe()) ;
    return ;
  }
  else {

  
    cppFile << "    patReal weight = observation->weight ;" << endl ;
    
    unsigned long R = (patModelSpec::the()->isMixedLogit()) 
      ? patModelSpec::the()->getAlgoNumberOfDraws() 
      : 1 ;
    
    if (patModelSpec::the()->applySnpTransform()) {
      err = new patErrMiscError("Not yet implemented") ;
      WARNING(err->describe()) ;
      return ;
    }
    
    if (R > 1) {
      cppFile << "    patReal probaPerDraw(0.0) ;" << endl ;
      cppFile << "    patReal sumOfProba(0.0) ;" << endl ;
    }
    else {
      cppFile << "    patReal logProbaForOnlyDraw(0.0) ;" << endl ;
    }

    if (derivatives) {
      if (R > 1) {
	cppFile << "    trVector gradientPerDraw(" << K << ",0.0) ;" << endl ;
	cppFile << "    trVector sumOfGradient(" << K << ",0.0) ;" << endl ;
      }
      else {
	cppFile << "    trVector gradientLogOnlyDraw(" << K << ",0.0) ;" << endl ;
      }
    }

    if (R > 1) {
      cppFile << "for (unsigned long drawNumber = 1 ; drawNumber <= " << R << " ; ++drawNumber) {" << endl ;
    }

    generateCppCodePerDraw(cppFile,
                           (R == 1),
 			   derivatives,
 			   secondDerivatives,
 			   err)  ;


    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }

    if (R > 1) {
      cppFile << "    sumOfProba += probaPerDraw ;" << endl ;
      if (derivatives) {
	cppFile << "    sumOfGradient += gradientPerDraw ;" << endl ;
      }
      
    }

    // Close the loop on draws
    if (R > 1) {
      cppFile << "    } // for (unsigned long drawNumber = 1 ; drawNumber <= " << R << " ; ++drawNumber) " << endl ;
    }

    if (R > 1) {
      cppFile << "    logProbaOneObs =  weight * (log(sumOfProba) - log(patReal(" << R << "))) ;" << endl ;
      if (derivatives) {
	cppFile << "gradientLogOneObs = (weight / sumOfProba) * sumOfGradient ;" ;
      }
    }
    else {
      cppFile << "    logProbaOneObs =  (weight * logProbaForOnlyDraw) ;" << endl ;
      if (derivatives) {
	cppFile << "	gradientLogOneObs = weight * gradientLogOnlyDraw ;" ;
      }
    }
    
  }
  
  cppFile << "//end of patProbaModel" << endl ;
  cppFile << "//////////////////////////////////" << endl ;
  
}


