//-*-c++-*------------------------------------------------------------
//
// File name : patProbaGevModel.cc
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
#include "patProbaGevModel.h"
#include "patMath.h"
#include "patGEV.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patRandomDraws.h"
#include "patValueVariables.h"
#include "patTimer.h"

patProbaGevModel::patProbaGevModel(patGEV* _gevFunction,
				   patUtility* aUtility) :
  patProbaModel(aUtility),
  gevFunction(_gevFunction),
  minExpArgument(1.0),
  maxExpArgument(1.0),
  overflow(0),
  underflow(0) {
  J = patModelSpec::the()->getNbrAlternatives() ;
  V.resize(J) ;
  term.resize(J) ;
  termDeriv.resize(J) ;
  expV.resize(J) ;
  G.resize(J) ;
  DG.resize(J,vector<patReal>(J)) ;
  dgdAlpha.resize(J) ;
  patVariables utilDerivatives ;
  correctForSelectionBias = patModelSpec::the()->correctForSelectionBias() ;
  corrBiasIndices.resize(J,patBadId) ;
}

patProbaGevModel::~patProbaGevModel() {

}



patReal patProbaGevModel::evalProbaPerDraw(patBoolean logOfProba,
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
    err = new patErrMiscError("SNP not yet implemented for GEV models") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }


  //  patBetaLikeParameter* MGEV = patModelSpec::the()->getMGEVParameter() ;

  x = &individual->attributes ;
  index = individual->choice ;
  weight = individual->weight ;
  indivId = individual->id ;

  J = patModelSpec::the()->getNbrAlternatives() ;
  nBeta = beta->size() ;

    
  static patBoolean first = patTRUE ;
  if (first) {
    utilDerivatives.resize(beta->size()) ;
    DV.resize(nBeta,vector<patReal>(J)) ;
    term.resize(J) ;
    termDeriv.resize(J) ;
    if (correctForSelectionBias) {
      for ( j = 0 ; j < J ; ++j) {
	indexCorr = 
	  patModelSpec::the()->getIdOfSelectionBiasParameter(j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	return patReal() ;
	}
	corrBiasIndices[j] = indexCorr ;
      }
    }

    first = patFALSE ;
  }


  altId = patModelSpec::the()->getAltInternalId(index,err) ;
  if (err != NULL) {
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


  //  For each available alternative compute j , V_j, e^{V_j}, et G_j 


  maxUtility = -patMaxReal ;
  
  for ( j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {

      V[j] = utility->computeFunction(indivId,
				      drawNumber,
				      j,
				      beta,
				      x,
				      err) ;
      
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }

      // Transform the utilities for the Generalized MEV model
//       if (MGEV != NULL) {
// 	patReal mgevValue = MGEV->estimated ;
// 	patReal theTerm = 1.0 + mgevValue * V[j] ;
// 	if (theTerm <= 0) {
// 	  stringstream str ;
// 	  str << "The sign condition for the generalized extreme value model is not verified: param = " << mgevValue << ", utility = " << V[j] ;
// 	  err = new patErrMiscError(str.str()) ;
// 	  WARNING(err->describe()) ;
// 	  return patReal() ;
// 	}
// 	else {
// 	  V[j] = log(theTerm) / mgevValue ;
// 	}
//       }

      if (V[j] > maxUtility) {
	maxUtility = V[j] ;
      }
    }
  }

  
  for ( j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {
      V[j] -= maxUtility;
      
      if (scale*V[j] > maxExpArgument) {
	maxExpArgument = scale*V[j] ;
      }
      if (scale*V[j] < minExpArgument) {
	minExpArgument = scale*V[j] ;
      }
      if (scale*V[j] >= patLogMaxReal::the()) {
	WARNING("Overflow for alt " << j << ": exp(" << scale*V[j] << ")") ;
	++overflow ;
	expV[j] = patMaxReal ;
      }
      else {
	expV[j] = exp(scale*V[j]) ;
  	if (expV[j] <= patMinReal) {
	  ++underflow ;
	}
      }
    }							    
  }


  // Compute the derivatives of the  GEV function

  gevFunction->compute(&expV,
		       parameters,
		       mu,
		       individual->availability,
		       !noDerivative,
		       err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
		       

  // Compute Gj et Delta

  Delta = 0.0 ;

  patReal maxTerm = -patMaxReal ;


  for ( j = 0 ; j < J ; ++j) {
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
      
      patReal logG = log(G[j]) ;
      if (!isfinite(logG)) {
	term[j] = -patMaxReal  ;
      }
      else {
	term[j] = V[j] + logG ;
      }
      if (!isfinite(term[j])) {
	WARNING("Numerical problem: " <<  V[j] << "+" << log(G[j])) ;
	++overflow ;
      }
      
      if (correctForSelectionBias && corrBiasIndices[j] != patBadId) {
	term[j] += (*beta)[corrBiasIndices[j]] ; 
      }
      
      if (term[j] > maxTerm) {
	maxTerm = term[j] ;
      }
      
    }
  }
  
  for ( j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {

      term[j] -= maxTerm ;
      if (!isfinite(term[j])) {
	//WARNING("Numerical problem: " <<  V[j] << "+" << log(G[j])) ;
	++overflow ;
	Delta = patMaxReal ;
      }
      else if (term[j] >= patMaxReal - Delta) {
	//WARNING("Numerical problem: " << Delta << "+" << term[j] << " too big") ; 
	++overflow ;
	Delta = patMaxReal ;
      }
      else {
	Delta += exp(term[j]) ;
      }
    }
  }

  if (!patFinite(Delta)) {
    ++overflow ;
    Delta = patMaxReal ;
  }

  result = term[altId] - log(Delta) ;

  if (!patFinite(result)) {
    ++underflow ;
    result = -patMaxReal ;
  }

  if (isfinite(result) == 0) {
    stringstream str ;
    str << "Cannot compute " << term[altId] << "-" << log(Delta) ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    (*success) = patFALSE ;
    return patReal() ;
  }

  theResult = (logOfProba) ? result : exp(result) ;

  if (!noDerivative) {
    // Derivatives with respect to beta


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

    for ( i = 0 ; i < J ; ++i) {
      if (individual->availability[i]) {


	fill(utilDerivatives.begin(),utilDerivatives.end(),0.0) ;
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
	
// 	if (MGEV != NULL) {
// 	  patReal mgevValue = MGEV->estimated ;
// 	  patReal theTerm = 1.0 + mgevValue * V[i] ;
// 	  // Transform the derivatives for the generalized MEV model
// 	  for ( k = 0 ; k < nBeta ; ++k) {
// 	    if (compBetaDerivatives[k]) {
// 	      utilDerivatives[k] /= theTerm ;
// 	      if (!isfinite(utilDerivatives[k])) {
// 		WARNING("Numerical problem for the MGEV model: division by " << theTerm) ;
// 		utilDerivatives[k] = patMaxReal ;
// 	      }
// 	    }	    
// 	  }	
// 	}

	for ( k = 0 ; k < nBeta ; ++k) {
	  if (compBetaDerivatives[k]) {
	    DV[k][i] = scale * utilDerivatives[k] ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return patReal()  ;
	    }
	  }
	}
	for ( j = i ; j < J ; ++j) {
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


    for ( k = 0 ; k < nBeta ; ++k) {
      
      if (compBetaDerivatives[k]) {
	
	patReal dDeltadBeta(0.0) ;

	for ( j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {

	    termDeriv[j] = DV[k][j] ;
	    
	    patReal dGdBeta(0.0) ;
	    
	    for ( n = 0 ; n < J ; ++n) {
	      if (individual->availability[n]) {
		dGdBeta += DG[j][n] * expV[n] * scale * DV[k][n] ;
	      }
	    }

	    patReal frac = dGdBeta / G[j] ;
	    if (!isfinite(frac)) {
	      if (dGdBeta > 0) {
		termDeriv[j] = patMaxReal ;
	      }
	      else if (dGdBeta < 0) {
		termDeriv[j] = -patMaxReal ;
	      }
	    }
	    else {
	      termDeriv[j] += frac ;
	    }
	    
	    if (corrBiasIndices[j] == k) {
	      termDeriv[j] += 1.0 ;
	    }
	    
	    dDeltadBeta += exp(term[j]) * termDeriv[j] ; 
	  }
	}

	if (logOfProba) {
	  (*betaDerivatives)[k] +=  termDeriv[altId] - dDeltadBeta / Delta ; 
	}
	else {
	  (*betaDerivatives)[k] += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ; 
	}
	
	if (isfinite((*betaDerivatives)[k]) == 0){
	  (*success) = patFALSE ;
	  (*betaDerivatives)[k] = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
      }
    }


    
      
    // Derivatives with respect to the GEV parameters

    nParam = compParamDerivatives.size() ;
    if (nParam != gevFunction->getNbrParameters()) {
      stringstream str ;
      str << "There should be " << gevFunction->getNbrParameters() 
	  << " parameters instead of " << nParam ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    paramDerivatives->resize(nParam) ;

    for ( k = 0 ; k < nParam ; ++k) {
      if (compParamDerivatives[k]) {

	patReal dDeltadBeta(0.0) ;

	for ( j = 0 ; j < J ; ++j) {

	  if (individual->availability[j]) {
	    deriv =  gevFunction->getSecondDerivative_param(j,
							    k,
							    &expV,
							    parameters,
							    mu,
							    individual->availability,							    err) ;
	    

	    patReal frac = deriv / G[j] ;
	    if (!isfinite(frac)) {
	      if (deriv > 0) {
		termDeriv[j] = patMaxReal ;
	      }
	      else if (deriv < 0) {
		termDeriv[j] = -patMaxReal ;
	      }
	    }
	    else {
	      termDeriv[j] = frac ;
	    }
	    
	    dDeltadBeta += exp(term[j]) * termDeriv[j] ; 
	  }
	}

	if (logOfProba) {
	  (*paramDerivatives)[k] +=  termDeriv[altId] - dDeltadBeta / Delta ; 
	}
	else {
	  (*paramDerivatives)[k] += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ; 
	}
	
	if (isfinite((*paramDerivatives)[k]) == 0) {
	  (*success) = patFALSE ;
	  (*paramDerivatives)[k] = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
      }
    }

    // Derivatives with respect to the mu parameter

    if (compMuDerivative) {

      patReal dDeltadBeta(0.0) ;

      for ( j = 0 ; j < J ; ++j) {
	if (individual->availability[j]) {
	  deriv =
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
	  
	  
	  patReal frac = deriv / G[j] ;
	  if (!isfinite(frac)) {
	    if (deriv > 0) {
	      termDeriv[j] = patMaxReal ;
	    }
	    else if (deriv < 0) {
	      termDeriv[j] = -patMaxReal ;
	    }
	  }
	  else {
	    termDeriv[j] = frac ;
	  }
	    
	  dDeltadBeta += exp(term[j]) * termDeriv[j] ; 
	}
      }

      if (logOfProba) {
	(*muDerivative) += termDeriv[altId] - dDeltadBeta / Delta ;  
      }
      else {
	(*muDerivative) += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ;
      }
      if (isfinite(*muDerivative) == 0) {
	(*success) = patFALSE ;
	*muDerivative = patMaxReal ;
	++overflow ;
	return patMaxReal ;
      }

    }
    
    // Derivative with respect to the scale parameter

    if (compScaleDerivative) {

      patReal dDeltadBeta(0.0) ;

      for ( j = 0 ; j < J ; ++j) {
	 
	if (individual->availability[j]) {
	  patReal dGdLambda(0.0) ;
	  
	  for (n = 0 ; n < J ; ++n) {
	    if (individual->availability[n]) {
	      dGdLambda += DG[j][n] * expV[n] * V[n] ;
	    }
	  }

	  patReal frac = deriv / G[j] ;
	  if (!isfinite(frac)) {
	    if (deriv > 0) {
	      termDeriv[j] = patMaxReal ;
	    }
	    else if (deriv < 0) {
	      termDeriv[j] = -patMaxReal ;
	    }
	  }
	  else {
	    termDeriv[j] = V[j] + frac ;
	  }

	  dDeltadBeta += exp(term[j]) * termDeriv[j] ; 
	  
	}
      }
      if (logOfProba) {
	(*scaleDerivative) += termDeriv[altId] - dDeltadBeta / Delta ;  
      }
      else {
	(*scaleDerivative) += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ;
      }
      if (isfinite(*scaleDerivative) == 0) {
	(*success) = patFALSE ;
	*scaleDerivative = patMaxReal ;
	++overflow ;
	return patMaxReal ;
      }
    }
  }

  (*success) = patTRUE ;

  return theResult ;
}






void patProbaGevModel::setGevFunction(patGEV* gevPtr) {
  gevFunction = gevPtr ;
}

patString patProbaGevModel::getModelName(patError*& err) {
  if (gevFunction == NULL) {
    err = new patErrNullPointer("patGEV") ;
    WARNING(err->describe()) ;
    return patString();
  }
  return gevFunction->getModelName() ;
}


patReal patProbaGevModel::getMinExpArgument() {
  return minExpArgument ;
}
patReal patProbaGevModel::getMaxExpArgument() {
  return maxExpArgument ;
}

unsigned long patProbaGevModel::getUnderflow() {
  return underflow ;
}

unsigned long patProbaGevModel::getOverflow() {
  return overflow ;
}

patString patProbaGevModel::getInfo() {
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


void patProbaGevModel::generateCppCodePerDraw(ostream& cppFile,
					      patBoolean logOfProba,
					      patBoolean derivatives, 
					      patBoolean secondDerivatives, 
					      patError*& err) {

  unsigned long K = patModelSpec::the()->getNbrNonFixedParameters() ;
  cppFile << "    //////////////////////////////////" << endl ;
  cppFile << "    // Code generated in patProbaGevModel" << endl ;

  if (!patModelSpec::the()->groupScalesAreAllOne()) {
    cppFile << "  unsigned long scaleIndex = groupIndex[observation->group] ;" << endl ;
    cppFile << "  patReal defaultScale = scalesPerGroup[observation->group] ;" ;
  }
  utility->genericCppCode(cppFile,derivatives,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  cppFile << "  for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
  cppFile << "    if (observation->availability[j]) {" << endl ;
  cppFile << "  if (V[j] >= " << patLogMaxReal::the() << ") {" << endl ;
  cppFile << "    expV[j] = " << patMaxReal << " ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  else {" << endl ;
  cppFile << "    expV[j] = exp(V[j]) ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  } //if (observation->availability[j]  " << endl ;
  cppFile << "  } //for ( j = 0 ; j < "<< J <<" ; ++j) " << endl ;


  gevFunction->generateCppCode(cppFile,derivatives,err) ;

  cppFile << "  // Compute Gj et Delta" << endl ;

  cppFile << "  patReal Delta(0.0) ;" << endl ;

  cppFile << "  patReal maxTerm = -" << patMaxReal << " ;" << endl ;
  cppFile << "  vector<patReal> term(" << J << ") ;" << endl ;

  for ( j = 0 ; j < J ; ++j) {
    cppFile << "    if (observation->availability[" << j << "]) {" << endl ;
      
    cppFile << "    patReal logG = log(firstDeriv_xi[" <<j << "]) ;" << endl ;
    cppFile << "    if (!isfinite(logG)) {" << endl ;
    cppFile << "      term[" << j << "] = -" << patMaxReal << " ;" << endl ;
    cppFile << "    }" << endl ;
    cppFile << "    else {" << endl ;
    cppFile << "      term[" << j << "] = V[" << j << "] + logG ;" << endl ;
    cppFile << "    }" << endl ;
      
    cppFile << "    if (term[" << j << "] > maxTerm) {" << endl ;
    cppFile << "      maxTerm = term[" << j << "] ;" << endl ;
    cppFile << "    }" << endl ;
    
    cppFile << "    } // if (observation->availability[" << j << "])" << endl ;
  }
  
  for ( j = 0 ; j < J ; ++j) {
    cppFile << "    if (observation->availability[" << j << "]) {" << endl ;

    cppFile << "      term[" << j << "] -= maxTerm ;" << endl ;
    cppFile << "      if (!isfinite(term[" << j << "])) {" << endl ;
    cppFile << "	Delta = " << patMaxReal << " ;" << endl ;
    cppFile << "      }" << endl ;
    cppFile << "      else if (term[" << j << "] >= " << patMaxReal << " - Delta) {" << endl ;
    cppFile << "	Delta = " << patMaxReal << " ;" << endl ;
    cppFile << "      }" << endl ;
    cppFile << "      else {" << endl ;
    cppFile << "	Delta += exp(term[" << j << "]) ;" << endl ;
    cppFile << "      }" << endl ;
      cppFile << "    } // if (observation->availability[" << j << "])" << endl ;
  }

  cppFile << "  if (!patFinite(Delta)) {" << endl ;
  cppFile << "    Delta = " << patMaxReal << " ;" << endl ;
  cppFile << "  }" << endl ;

  cppFile << "  patReal result = term[altIndex[observation->choice]] - log(Delta) ;" << endl ;

  cppFile << "  if (!patFinite(result)) {" << endl ;
  cppFile << "    result = -" << patMaxReal << " ;" << endl ;
  cppFile << "  }" << endl ;

  if (logOfProba) {
    cppFile << "    logProbaForOnlyDraw = result ;" << endl ;
  }
  else {
    cppFile << "    probaPerDraw = exp(result) ;" << endl ;
  }

  if (derivatives) {
    stringstream whenNoLog ;
    if (!logOfProba) {
      whenNoLog << "* probVec[altIndex[observation->choice]] " ;
    }
    cppFile << "	for (unsigned long k = 0 ; k < " << K << " ; ++k) {" << endl ;
    
    stringstream gradientName ;
    if (logOfProba) {
      gradientName << "gradientLogOnlyDraw" ;
    }
    else {
      gradientName << "gradientPerDraw" ;
    }

    // Derivatives with respect to beta

    cppFile << "    vector<patReal> termDeriv(" << J << ") ;" << endl ;
    cppFile << "    patReal dDeltadBeta ;" << endl ;
    cppFile << "    patReal dGdBeta ;" << endl ;
    patIterator<patBetaLikeParameter>* iter = patModelSpec::the()->createAllBetaIterator() ;
    for (iter->first() ;
	 !iter->isDone() ;
	 iter->next()) {
      patBetaLikeParameter theBeta = iter->currentItem() ;
      if (!theBeta.isFixed) {
	unsigned long k = theBeta.index ;
	
	cppFile << "      dDeltadBeta = 0.0 ;" << endl ;
	for ( j = 0 ; j < J ; ++j) {
	  cppFile << "      if (observation->availability[" << j << "]) {" << endl ;
	  
	  cppFile << "	termDeriv[" << j << "] = DV[" << k << "][" << j << "] ;" << endl ;
	  
	  cppFile << "	dGdBeta = 0.0 ;" << endl ;
	  
	  for ( n = 0 ; n < J ; ++n) {
	    cppFile << "	  if (observation->availability[" << n << "]) {" << endl ;
	    cppFile << "	    dGdBeta += secondDeriv_xi_xj[" << j << "][" << n << "] * expV[" << n << "] * " ;
	    cppFile << "(scaleIndex == " << patBadId << ")?defaultScale:(*x)[scaleIndex]" ;
	    cppFile << " * DV[" << k << "][" << n << "] ;" << endl ;
	    cppFile << "	  } // if (observation->availability[n]) " << endl ;
	  }
	  
	  cppFile << "	patReal frac = dGdBeta / G[" << j << "] ;" << endl ;
	  cppFile << "	if (!isfinite(frac)) {" << endl ;
	  cppFile << "	  if (dGdBeta > 0) {" << endl ;
	  cppFile << "	    termDeriv[" << j << "] = " << patMaxReal << " ;" << endl ;
	  cppFile << "	  }" << endl ;
	  cppFile << "	  else if (dGdBeta < 0) {" << endl ;
	  cppFile << "	    termDeriv[" << j << "] = -" << patMaxReal << " ;" << endl ;
	  cppFile << "	  }" << endl ;
	  cppFile << "	}" << endl ;
	  cppFile << "	else {" << endl ;
	  cppFile << "	  termDeriv[" <<j << "] += frac ;" << endl ;
	  cppFile << "	}" << endl ;
	  
	  cppFile << "	dDeltadBeta += exp(term[" << j << "]) * termDeriv[" << j << "] ; " << endl ;
	  cppFile << "      } // if (observation->availability[" << j << "])" << endl ;
	  
	  if (logOfProba) {
	    cppFile << "	  " << gradientName.str() << "[k] +=  termDeriv[altIndex[observation->choice]] - dDeltadBeta / Delta ; " << endl ;
	  }
	  else {
	    cppFile << "	  "<< gradientName.str() <<"[k] += probaPerDraw * (termDeriv[altIndex[observation->choice]] - dDeltadBeta / Delta) ; " << endl ;
	  }
	  
	  cppFile << "	if (isfinite("<<gradientName.str()<<"[k]) == 0){" << endl ;
	  cppFile << "	  (*success) = patFALSE ;" << endl ;
	  cppFile << "	  "<<gradientName.str()<<"[k] = " << patMaxReal << " ;" << endl ;
	  cppFile << "	  return patMaxReal ;" << endl ;
	  cppFile << "	}" << endl ;
	}
      }
    } // for ( iter->first()... )
    

    iter = patModelSpec::the()->createAllModelIterator() ;
    for (iter->first() ;
	 !iter->isDone() ;
	 iter->next()) {
      patBetaLikeParameter theBeta = iter->currentItem() ;
      if (!theBeta.isFixed) {
	unsigned long k = theBeta.index ;      
	unsigned long paramId = theBeta.id ;      

	cppFile << "	patReal dDeltadBeta(0.0) ;" << endl ;

	for ( j = 0 ; j < J ; ++j) {

	  cppFile << "	  if (observation->availability[" << j << "]) {" << endl ;
	    
	    cppFile << "	    patReal frac = secondDeriv_xi_param[" << j << "][" << paramId << "]  / firstDeriv_xi[" << j << "] ;" << endl ;
	    cppFile << "	    if (!isfinite(frac)) {" << endl ;
	    cppFile << "	      if (deriv > 0) {" << endl ;
	    cppFile << "		termDeriv[" << j << "] = " << patMaxReal << " ;" << endl ;
	    cppFile << "	      }" << endl ;
	    cppFile << "	      else if (deriv < 0) {" << endl ;
	    cppFile << "		termDeriv[" << j << "] = -" << patMaxReal << " ;" << endl ;
	    cppFile << "	      }" << endl ;
	    cppFile << "	    } // if (!isfinite(frac)) " << endl ;
	    cppFile << "	    else {" << endl ;
	    cppFile << "	      termDeriv[" << j << "] = frac ;" << endl ;
	    cppFile << "	    }" << endl ;
	    
	    cppFile << "	    dDeltadBeta += exp(term[" << j << "]) * termDeriv[" << j << "] ; " << endl ;
	    cppFile << "	  } // if (observation->availability[" << j << "])" << endl ;
	}

	cppFile << "	if (logOfProba) {" << endl ;
	cppFile << "	  "<< gradientName.str() <<"[" << k << "] +=  termDeriv[altIndex[observation->choice]] - dDeltadBeta / Delta ; " << endl ;
	cppFile << "	}" << endl ;
	cppFile << "	else {" << endl ;
	cppFile << "	  " << gradientName.str() << "[" << k << "] += theResult * (termDeriv[altIndex[observation->choice]] - dDeltadBeta / Delta) ; " << endl ;
	cppFile << "	}" << endl ;
	
      }
    }

    // Derivatives with respect to the mu parameter

    patBetaLikeParameter mu = patModelSpec::the()->getMu(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
    if (!mu.isFixed) {


      cppFile << "      patReal dDeltadBeta(0.0) ;" << endl ;

      for ( j = 0 ; j < J ; ++j) {
	cppFile << "	if (observation->availability[" << j << "]) {" << endl ;
	cppFile << "	  patReal frac = muDerivative[" << j << "] / firstDeriv_xi[" << j << "] ;" << endl ;
	cppFile << "	if (!isfinite(frac)) {" << endl ;
	cppFile << "	  if (deriv > 0) {" << endl ;
	cppFile << "	    termDeriv[" << j << "] = " << patMaxReal << " ;" << endl ;
	cppFile << "	  }" << endl ;
	cppFile << "	  else if (deriv < 0) {" << endl ;
	cppFile << "	    termDeriv[" << j << "] = -" << patMaxReal << " ;" << endl ;
	cppFile << "	  }" << endl ;
	cppFile << "	}" << endl ;
	cppFile << "	else {" << endl ;
	cppFile << "	    termDeriv[" << j << "] = frac ;" << endl ;
	cppFile << "	}" << endl ;
	    
	cppFile << "	  dDeltadBeta += exp(term[" << j << "]) * termDeriv[" << j << "] ; " << endl ;
	  cppFile << "	} 	//if (observation->availability[" << j << "])" << endl ;
      }

      if (logOfProba) {
	cppFile << "	"<<gradientName.str() << "[" << mu.index << "] += termDeriv[altIndex[observation->choice]] - dDeltadBeta / Delta ;  " << endl ;
      }
      else {
	cppFile << "	" <<gradientName.str() << "[" << mu.index << "] += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ;" << endl ;
      }
    }
    
  
//     // Derivative with respect to the scale parameter

//     if (compScaleDerivative) {

//       patReal dDeltadBeta(0.0) ;

//       for ( j = 0 ; j < J ; ++j) {
	 
// 	if (individual->availability[j]) {
// 	  patReal dGdLambda(0.0) ;
	  
// 	  for (n = 0 ; n < J ; ++n) {
// 	    if (individual->availability[n]) {
// 	      dGdLambda += DG[j][n] * expV[n] * V[n] ;
// 	    }
// 	  }

// 	  patReal frac = deriv / G[j] ;
// 	  if (!isfinite(frac)) {
// 	    if (deriv > 0) {
// 	      termDeriv[j] = patMaxReal ;
// 	    }
// 	    else if (deriv < 0) {
// 	      termDeriv[j] = -patMaxReal ;
// 	    }
// 	  }
// 	  else {
// 	    termDeriv[j] = V[j] + frac ;
// 	  }

// 	  dDeltadBeta += exp(term[j]) * termDeriv[j] ; 
	  
// 	}
//       }
//       if (logOfProba) {
// 	(*scaleDerivative) += termDeriv[altId] - dDeltadBeta / Delta ;  
//       }
//       else {
// 	(*scaleDerivative) += theResult * (termDeriv[altId] - dDeltadBeta / Delta) ;
//       }
//       if (isfinite(*scaleDerivative) == 0) {
// 	(*success) = patFALSE ;
// 	*scaleDerivative = patMaxReal ;
// 	++overflow ;
// 	return patMaxReal ;
//       }
//     }
//   }

//   (*success) = patTRUE ;

//   return theResult ;



  cppFile << "    // End of code generated in patProbaGevModel" << endl ;
  cppFile << "    ////////////////////////////////////////////" << endl ;

  }
}
