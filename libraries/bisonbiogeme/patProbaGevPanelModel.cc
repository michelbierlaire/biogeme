//-*-c++-*------------------------------------------------------------
//
// File name : patProbaGevPanelModel.cc
// Author :    Michel Bierlaire
// Date :      Tue Jul 11 20:44:29 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patUtility.h"
#include "patModelSpec.h"
#include "patProbaGevPanelModel.h"
#include "patMath.h"
#include "patGEV.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patRandomDraws.h"
#include "patValueVariables.h"
#include "patTimer.h"

patProbaGevPanelModel::patProbaGevPanelModel(patGEV* _gevFunction,
				   patUtility* aUtility) :
  patProbaPanelModel(aUtility),
  gevFunction(_gevFunction),
  minExpArgument(1.0),
  maxExpArgument(1.0),
  overflow(0),
  underflow(0)  {

}

patProbaGevPanelModel::~patProbaGevPanelModel() {
  DELETE_PTR(betaDrawDerivatives) ;
  DELETE_PTR(paramDrawDerivatives) ;
  DELETE_PTR(muDrawDerivative) ;
  DELETE_PTR(scaleDrawDerivative) ;

}



patReal patProbaGevPanelModel::evalProbaPerObs(patObservationData* individual,
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

  //    patTimer::the()->tic(0) ;

  patValueVariables::the()->setAttributes(&(individual->attributes)) ;
  patValueVariables::the()->setRandomDraws(&(individual->draws[drawNumber-1])) ;


  vector<patObservationData::patAttributes>* x(&individual->attributes) ;
  unsigned long index = individual->choice ;
  //  patReal weight = individual->weight ;
  unsigned long indivId = individual->id ;

  //  DEBUG_MESSAGE("--> debug 1") ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  unsigned long J = patModelSpec::the()->getNbrAlternatives() ;

  if (utility == NULL) {
    err = new patErrNullPointer("patUtility") ;
    WARNING(err->describe()) ;
    return patReal() ;

  }

  if (!noDerivative) {
    if (betaDerivatives == NULL || paramDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    if (compMuDerivative && muDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    if (compScaleDerivative && scaleDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }

  if (beta == NULL || parameters == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }


  if (x == NULL) {
    err = new patErrNullPointer("vector<patVariables>") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (!noDerivative) {
    if (betaDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      return patReal()  ;
    }
    if (paramDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      return patReal()  ;
    }
    if (muDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      return patReal()  ;
    }
  }

  if (mu == NULL) {
    err = new patErrNullPointer("patReal") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (gevFunction == NULL) {
    err = new patErrNullPointer("patGEV") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  // Check if the sizes and index are compatibles

  //  DEBUG_MESSAGE("--> debug 2") ;
  unsigned long altId = patModelSpec::the()->getAltInternalId(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (x->size() != patModelSpec::the()->getNbrUsedAttributes()) {
    err = new patErrMiscError("Incompatible nbr of characteristics") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (individual->availability.size() != J) {
    err = new patErrMiscError("Incompatible nbr of availabilities") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (altId >= J) {
    err = new patErrOutOfRange<unsigned long>(altId,0,J-1) ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  // Check if the chosen alternative is available

  if (!individual->availability[altId]) {
    stringstream str ;
    str << "Alternative " << altId << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  //    patTimer::the()->toc(0) ;
  // At this point, everything seems to be ready to compute
  //  DEBUG_MESSAGE("--> debug 3") ;

  //  For each available alternative compute j , V_j, e^{V_j}, et G_j 

  patVariables V(J) ;
  patVariables expV(J) ;

  patReal maxUtility = -patMaxReal ;
  
  //  patTimer::the()->tic(2) ;

  for (unsigned long j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {

      //      patTimer::the()->tic(1) ;
    

      V[j] = utility->computeFunction(indivId,
				      drawNumber,
				      j,
				      beta,
				      x,
				      err) ;
      
      //      patTimer::the()->toc(1) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      if (V[j] > maxUtility) {
	maxUtility = V[j] ;
      }
    }
  }
  
  //  patTimer::the()->toc(2) ;

  //  patTimer::the()->tic(3) ;

  for (unsigned long j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {
      //  DEBUG_MESSAGE("--> debug 4") ;
      V[j] -= maxUtility;
      
      if (scale*V[j] > maxExpArgument) {
	maxExpArgument = scale*V[j] ;
      }
      if (scale*V[j] < minExpArgument) {
	minExpArgument = scale*V[j] ;
      }
      if (scale*V[j] >= patLogMaxReal::the()) {
  	DEBUG_MESSAGE("scale = " << scale) ;
  	DEBUG_MESSAGE("V["<<j<<"]=" << V[j]) ;
	WARNING("Overflow for alt " << j << ": exp(" << scale*V[j] << ")") ;
	++overflow ;
	expV[j] = patMaxReal ;
      }
      else {
	expV[j] = exp(scale*V[j]) ;
  	if (expV[j] <= patMinReal) {
	  //  	  WARNING("Underflow exp[" << scale << "*" << V[j] << "]=" << expV[j]) ; 
	  ++underflow ;
	}
      }
    }							    
  }

  //  patTimer::the()->toc(3) ;

  // Compute Gj et Delta

  //  DEBUG_MESSAGE("--> debug 5") ;
  vector<patReal> G(J) ;


  // Compute the derivatives of the  GEV function

  gevFunction->compute(&expV,
		       parameters,
		       mu,
		       individual->availability,
		       !noDerivative,
		       err) ;

  patReal Delta = 0.0 ;

  //  patTimer::the()->tic(4) ;

  for (unsigned long j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {

      G[j] = gevFunction->getDerivative_xi(j,
					   &expV,
					   parameters,
					   mu,
					   individual->availability,
					   err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal()  ;
      }
      
      
      patReal product = expV[j] * G[j] ;
      if (!isfinite(product)) {
	WARNING("Numerical problem: " << expV[j] << "*" << G[j]) ;
	++overflow ;
	Delta = patMaxReal ;
	break ;
      }
      else if (product >= patMaxReal - Delta) {
	WARNING("Numerical problem: " << Delta << "+" << product << " too big") ; 
	++overflow ;
	Delta = patMaxReal ;
	break ;
      }
      else {
	Delta += expV[j] * G[j] ;
      }
    }
  }

  //  patTimer::the()->toc(4) ;

  if (!patFinite(Delta)) {
    ++overflow ;
    Delta = patMaxReal ;
  }

  patReal result(0.0);
  if (G[altId] > patMinReal) {
    result = scale * V[altId] + log(G[altId]) - log(Delta) ;
  }
  else {
//      for (unsigned long j = 0 ; j < J ; ++j) {
//        DEBUG_MESSAGE("G["<< j << "]=" << G[j]) ;
//      }

//      DEBUG_MESSAGE("altId=" << altId) ; 
//      for (unsigned long zz = 0 ;
//  	 zz < parameters->size() ;
//  	 ++zz) {
//        DEBUG_MESSAGE("param[" << zz << "]=" << (*parameters)[zz]) ;
//      }
//      for (unsigned long zz = 0 ;
//  	 zz < beta->size() ;
//  	 ++zz) {
//        DEBUG_MESSAGE("beta[" << zz << "]=" << (*beta)[zz]) ;
//      }
//      WARNING("Chosen alt. " << patModelSpec::the()->getAltId(altId,err)
//  	    << " has zero probability") ;
    //    cout << "*" << flush ;

    ++underflow ;
    return 0.0 ;
  }

  if (!patFinite(result)) {
    ++underflow ;
    result = -patMaxReal ;
  }

  if (isfinite(result) == 0) {
    stringstream str ;
    str << "Cannot compute " << scale << "*" <<  V[altId] << "+log(" 
	<< G[altId] << ")-log(" << Delta << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    (*success) = patFALSE ;
    return patReal() ;
  }


  patReal theProba = exp(result) ;

  if (!noDerivative) {
    // Derivatives with respect to beta

    unsigned long nBeta = compBetaDerivatives.size() ;

    if (nBeta != patModelSpec::the()->getNbrTotalBeta()) {
      stringstream str ;
      str << "There should be " << patModelSpec::the()->getNbrTotalBeta() 
	  << " beta's instead of " << nBeta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    

    // DG will contain the second derivatives of G with respect to i and
    // j. 

    vector<vector<patReal> > DG(J,vector<patReal>(J)) ;
    
    vector<vector<patReal> > DV(nBeta,vector<patReal>(J)) ;

    //    patTimer::the()->tic(5) ;
    for (unsigned long i = 0 ; i < J ; ++i) {
      if (individual->availability[i]) {
	patVariables utilDerivatives(beta->size()) ;
	
	utility->computeBetaDerivative(indivId,
				       drawNumber,
				       i,
				       beta,
				       x,
				       &utilDerivatives,
				       err) ;
	
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}

	for (unsigned long k = 0 ; k < nBeta ; ++k) {
	  if (compBetaDerivatives[k]) {
	    DV[k][i] = scale * utilDerivatives[k] ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return patReal()  ;
	    }
	  }
	}
	for (unsigned long j = i ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    DG[j][i] =
	      DG[i][j] = gevFunction->getSecondDerivative_xi_xj(i,
							      j,
							      &expV,
							      parameters,
							      mu,
							      individual->availability,
							      err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return patReal()  ;
	    }
	  }
	}
      }
    }
    //    patTimer::the()->toc(5) ;
    //    patTimer::the()->tic(6) ;

    patReal lastTerm ;
    patReal dgAltIdj ;
    patReal dvkj ;
    patReal tmp ;
    patReal expvj ;
    for (unsigned long k = 0 ; k < nBeta ; ++k) {
      
      if (compBetaDerivatives[k]) {
	(*betaDerivatives)[k] += theProba * DV[k][altId] ;
	for (unsigned long j = 0 ; j < J ; ++j) {
	  expvj = expV[j] ;
	  dvkj = DV[k][j] ;
	  if (individual->availability[j]) {
	    lastTerm = 0.0 ;
	    for (unsigned long n = 0 ; n < J ; ++n) {
	      if (individual->availability[n]) {
		lastTerm += DG[j][n] * expV[n] * DV[k][n] ;
	      }
	    }

	    dgAltIdj  = DG[altId][j] ;
	    
	    if (patAbs(dgAltIdj) > patEPSILON && 
		patAbs(expvj) > patEPSILON &&
		patAbs(dvkj) > patEPSILON) {
	      (*betaDerivatives)[k] += theProba *
	        dgAltIdj * expvj * dvkj / G[altId] ;
	    }
	    tmp = dvkj * G[j] + lastTerm ;
	    if (patAbs(expvj) > patEPSILON &&
		patAbs(tmp) > patEPSILON) {
	      (*betaDerivatives)[k] -= theProba *
		expvj * tmp / Delta ;
	    }
	  }	
	}
	if (isfinite((*betaDerivatives)[k]) == 0){
	  (*success) = patFALSE ;
	  (*betaDerivatives)[k] = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
      }
    }
    //    patTimer::the()->toc(6) ;

      
    // Derivatives with respect to the GEV parameters

    unsigned long nParam = compParamDerivatives.size() ;
    if (nParam != gevFunction->getNbrParameters()) {
      stringstream str ;
      str << "There should be " << gevFunction->getNbrParameters() 
	  << " parameters instead of " << nParam ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    paramDerivatives->resize(nParam) ;

    for (unsigned long k = 0 ; k < nParam ; ++k) {
      if (compParamDerivatives[k]) {
	patReal deriv =
	  gevFunction->getSecondDerivative_param(altId,
						 k,
						 &expV,
						 parameters,
						 mu,
						 individual->availability,
						 err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal()  ;
	}
	if (patAbs(deriv) > patEPSILON) {
	  deriv /= G[altId] ;
	  (*paramDerivatives)[k] += theProba * deriv ; 
	}
	for (unsigned long j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    patReal dgdp = 
	      gevFunction->getSecondDerivative_param(j,
						     k,
						     &expV,
						     parameters,
						     mu,
						     individual->availability,
						     err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return patReal()  ;
	    }
	    if (patAbs(expV[j]) > patEPSILON && patAbs(dgdp) > patEPSILON) {
	      (*paramDerivatives)[k] -= theProba * expV[j] * dgdp / Delta ;	    
	      if (isfinite((*paramDerivatives)[k]) == 0) {
		DEBUG_MESSAGE("expV[j]=" << expV[j]) ;
		DEBUG_MESSAGE("dgdp=" << dgdp) ;
		DEBUG_MESSAGE("Delta=" << Delta) ;
	      }
	    }
	  }
	}
	if (isfinite((*paramDerivatives)[k]) == 0) {
	  DEBUG_MESSAGE("Numerical problem with (*paramDerivatives)[" << k << "]") ;
	  (*success) = patFALSE ;
	  (*paramDerivatives)[k] = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
      }
    }

    // Derivatives with respect to the mu parameter

    if (compMuDerivative) {
      patReal deriv =
	gevFunction->getSecondDerivative_xi_mu(altId,
					    &expV,
					    parameters,
					    mu,
					    individual->availability,
					    err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal()  ;
      }
      deriv /= G[altId] ;
      (*muDerivative) += theProba * deriv ; 
      for (unsigned long j = 0 ; j < J ; ++j) {
	if (individual->availability[j]) {
	  patReal dgdp = 
	    gevFunction->getSecondDerivative_xi_mu(j,
						&expV,
						parameters,
						mu,
						individual->availability,
						err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal()  ;
	  }

	  (*muDerivative) -= theProba * expV[j] * dgdp / Delta ;	    
	}
      }
      if (isfinite(*muDerivative) == 0) {
	DEBUG_MESSAGE("Error in deriv mu") ;
	(*success) = patFALSE ;
	*muDerivative = patMaxReal ;
	++overflow ;
	return patMaxReal ;
      }

    }
    
    // Derivative with respect to the scale parameter

    if (compScaleDerivative) {

      patVariables dgdAlpha(J,0.0) ;
      patReal dDeltadAlpha = 0.0 ;
      for (unsigned long j = 0 ; j < J ; ++j) {
	if (individual->availability[j]) {
	  for (unsigned long k = 0 ; k < J ; ++k) {
	    if (individual->availability[k]) {
	      dgdAlpha[j] += V[k] * expV[k] * 
		gevFunction->getSecondDerivative_xi_xj(j,
						       k,
						       &expV,
						       parameters,
						       mu,
						       individual->availability,
						       err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return patReal()  ;
	      }
	    }
	  }
	  dDeltadAlpha += V[j] * expV[j] * G[j] + expV[j] * dgdAlpha[j] ;
	}
      }

      (*scaleDerivative) += theProba * V[altId] ;
      if (patAbs(dgdAlpha[altId]) > patEPSILON) {
	(*scaleDerivative) += theProba * dgdAlpha[altId] / G[altId] ;
      }
      if (patAbs(dDeltadAlpha) > patEPSILON) {
	(*scaleDerivative) -= theProba * dDeltadAlpha / Delta ;
      }
      if ((*scaleDerivative) > 1.0e10) {
	DEBUG_MESSAGE("Delta = " << Delta) ;
	DEBUG_MESSAGE("G[" << altId << "]=" << G[altId]) ;
      }
      if (isfinite((*scaleDerivative)) == 0) {
	DEBUG_MESSAGE("Delta = " << Delta) ;
	DEBUG_MESSAGE("G[" << altId << "]=" << G[altId]) ;
	(*success) = patFALSE ;
	(*scaleDerivative) = patMaxReal ;
	++overflow ;
	return patMaxReal ;
      }
    }
  }

  (*success) = patTRUE ;

  return theProba ;
}






void patProbaGevPanelModel::setGevFunction(patGEV* gevPtr) {
  gevFunction = gevPtr ;
}

patString patProbaGevPanelModel::getModelName(patError*& err) {
  if (gevFunction == NULL) {
    err = new patErrNullPointer("patGEV") ;
    WARNING(err->describe()) ;
    return patString();
  }
  patString res =  gevFunction->getModelName() + " with panel data" ;
  return res ;
}


patReal patProbaGevPanelModel::getMinExpArgument() {
  return minExpArgument ;
}
patReal patProbaGevPanelModel::getMaxExpArgument() {
  return maxExpArgument ;
}

unsigned long patProbaGevPanelModel::getUnderflow() {
  return underflow ;
}

unsigned long patProbaGevPanelModel::getOverflow() {
  return overflow ;
}

patString patProbaGevPanelModel::getInfo() {
  stringstream str ;
  str << "The minimum argument of exp was " << getMinExpArgument() << endl ;
  if (getUnderflow() > 0) {
    str << "Underflows: " << getUnderflow() << endl ;
  }
  if (getOverflow() > 0) {
    str << "Overflows:  " << getOverflow() << endl ;
  }
  return patString(str.str()) ;

}


void patProbaGevPanelModel::generateCppCodePerObs(ostream& cppFile,
						   patBoolean derivatives, 
						   patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return  ;
}
