//-*-c++-*------------------------------------------------------------
//
// File name : patProbaMnlModel.cc
// Author :    Michel Bierlaire
// Date :      Tue Aug 26 14:40:25 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patUtility.h"
#include "patModelSpec.h"
#include "patProbaMnlModel.h"
#include "patMath.h"
#include "patGEV.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patRandomDraws.h"
#include "patValueVariables.h"
#include "patTimer.h"
#include "patSecondDerivatives.h"

patProbaMnlModel::patProbaMnlModel(patUtility* aUtility) :
patProbaModel(aUtility), minExpArgument(1.0),maxExpArgument(1.0),overflow(0),underflow(0)  {

  J = patModelSpec::the()->getNbrAlternatives() ;
  V.resize(J) ;
  expV.resize(J) ;
  scaleMuV.resize(J) ;
  probVec.resize(J) ;

}

patProbaMnlModel::~patProbaMnlModel() {


}



patReal patProbaMnlModel::evalProbaPerDraw(patBoolean logOfProba,
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


//   if (secondDeriv == NULL) {
//     DEBUG_MESSAGE("SECOND DERIV = NULL") ;
//   }
//   else {
//     DEBUG_MESSAGE("SECOND DERIV != NULL") ;
//   }
  static patBoolean first = patTRUE ;
  if (first) {
    nBeta = beta->size() ;
    DV.resize(nBeta,vector<patReal>(J)) ;
    utilDerivatives.resize(J,patVariables(nBeta)) ;
    term.resize(nBeta) ;
    first = patFALSE  ;
  }


  //    patTimer::the()->tic(0) ;

  x = &individual->attributes ;
  index = individual->choice ;
  weight = individual->weight ;
  indivId = individual->id ;

  //    //  DEBUG_MESSAGE("--> debug 1") ;
  //    if (err != NULL) {
  //      WARNING(err->describe()) ;
  //      return patReal() ;
  //    }

  //    if (success == NULL) {
  //      err = new patErrNullPointer("patBoolean") ;
  //      WARNING(err->describe()) ;
  //      return patReal() ;
  //    }


  if (utility == NULL) {
    err = new patErrNullPointer("patUtility") ;
    WARNING(err->describe()) ;
    return patReal() ;

  }

  altId = patModelSpec::the()->getAltInternalId(index,err) ;

  //  For each available alternative compute j , V_j, e^{V_j}, et G_j 


  maxUtility = -patMaxReal ;
  
  //  patTimer::the()->tic(2) ;

  for (j = 0 ; j < J ; ++j) {
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

  sumExp = 0.0 ;

  for ( j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {
      V[j] -= maxUtility;
      scaleMuV[j] = scale*(*mu)*V[j] ;
      if (scaleMuV[j] > maxExpArgument) {
	maxExpArgument = scaleMuV[j] ;
      }
      if (scaleMuV[j] < minExpArgument) {
	minExpArgument = scaleMuV[j] ;
      }
      if (scaleMuV[j] >= patLogMaxReal::the()) {
	WARNING("Overflow for alt " << j << ": exp(" << scaleMuV[j] << ")") ;
	++overflow ;
	expV[j] = patMaxReal ;
      }
      else {
	expV[j] = exp(scaleMuV[j]) ;
	if (expV[j] <= patMinReal) {
	  //  	  WARNING("Underflow exp[" << *mu << "*" << scale << "*" << V[j] << "]=" << expV[j]) ; 
	  ++underflow ;
	}
      }
      sumExp += expV[j] ;
    }							    
  }

  //  patTimer::the()->toc(3) ;

  lsumExp = log(sumExp) ;

  if (noDerivative) {

    result = scaleMuV[altId] - lsumExp;
    if (!patFinite(result)) {
      ++underflow ;
      result = -patMaxReal ;
    }
    
    if (isfinite(result) == 0) {
      stringstream str ;
      str << "Cannot compute " << (*mu) << "*" <<  scale << "*" <<  V[altId] << "-log(" 
	  << sumExp << ")" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      (*success) = patFALSE ;
      return patReal() ;
    }
    
    if (logOfProba) {
      return result ;
    }
    else {
      return exp(result) ;
    }
  }
  else {
    for (j = 0 ; j < J ; ++j) {
      if (individual->availability[j]) {
	tmp = scaleMuV[j] - lsumExp;
	if (!patFinite(tmp)) {
	  ++underflow ;
	  tmp = -patMaxReal ;
	}
	
	if (isfinite(tmp) == 0) {
	  stringstream str ;
	  str << "Cannot compute " << (*mu) << "*" << scale << "*" <<  V[altId] << "-log(" 
	      << sumExp << ")" ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(err->describe()) ;
	  (*success) = patFALSE ;
	  return patReal() ;
	}
	probVec[j] = exp(tmp) ;
      }
    }
  }

  if (!noDerivative) {

    if (logOfProba) { //  if (logOfProba) 
      if (snpTerms) {
	err = new patErrMiscError("SNP terms are meaningless when no random parameter is present") ;
	WARNING(err->describe()) ;
	return patMaxReal ;
      }
      // Derivatives with respect to beta


      if (nBeta != patModelSpec::the()->getNbrTotalBeta()) {
	stringstream str ;
	str << "There should be " << patModelSpec::the()->getNbrTotalBeta() 
	    << " beta's instead of " << nBeta ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
    
      
      for (i = 0 ; i < J ; ++i ) {
	if (individual->availability[i]) {
	
	  fill(utilDerivatives[i].begin(),utilDerivatives[i].end(),0.0) ;
	  utility->computeBetaDerivative(indivId,
					 drawNumber,
					 i,
					 beta,
					 x,
					 &(utilDerivatives[i]),
					 err) ;
	
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }

	  for ( k = 0 ; k < nBeta ; ++k) {
	    if (compBetaDerivatives[k]) {
	      DV[k][i] = scale * utilDerivatives[i][k] ;
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

	  (*betaDerivatives)[k] += (*mu) * DV[k][altId] ;
	  for ( j = 0 ; j < J ; ++j) {
	    if (individual->availability[j]) {
	      (*betaDerivatives)[k] -= 
		(*mu) * probVec[j] * DV[k][j] ;
	    }
	  }
	}
      }

      // Derivatives with respect to the mu parameter

      if (compMuDerivative) {

	(*muDerivative) += scale * V[altId] ;
	for ( j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {

	    (*muDerivative) -= 
	      probVec[j] * scale * V[j] ;
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

	(*scaleDerivative) += (*mu) * V[altId] ;
	for ( j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    (*scaleDerivative) -= (*mu) * probVec[j] * V[j] ;
	  }
	}
	if (isfinite((*scaleDerivative)) == 0) {
	  (*success) = patFALSE ;
	  (*scaleDerivative) = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
      }

    }
    else {
      // Derivatives with respect to beta

   
      //    patTimer::the()->tic(5) ;
      for ( i = 0 ; i < J ; ++i ) {
	if (individual->availability[i]) {
	
	  fill(utilDerivatives[i].begin(),utilDerivatives[i].end(),0.0) ;
	  utility->computeBetaDerivative(indivId,
					 drawNumber,
					 i,
					 beta,
					 x,
					 &(utilDerivatives[i]),
					 err) ;
	
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }

	  for ( k = 0 ; k < nBeta ; ++k) {
	    if (compBetaDerivatives[k]) {
	      DV[k][i] = scale * utilDerivatives[i][k] ;
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

      for ( k = 0 ; k < nBeta ; ++k) {
	if (compBetaDerivatives[k]) {
	  theDeriv = 0.0 ;
	  theDeriv += (*mu) * DV[k][altId] * probVec[altId] ;
	  for ( j = 0 ; j < J ; ++j) {
	    if (individual->availability[j]) {
	      theDeriv -= 
		(*mu) * probVec[altId] * probVec[j] * DV[k][j] ;
	    }
	  }
	  if (snpTerms) {
	    (*betaDerivatives)[k] += theDeriv * factorForDerivOfSnpTerms ;
	  }
	  else {
	    (*betaDerivatives)[k] += theDeriv ;
	  }
	}
      }

      // Derivatives with respect to the mu parameter

      if (compMuDerivative) {
	theDeriv = 0.0 ;
	theDeriv += probVec[altId] * scale * V[altId] ;
	for ( j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {

	    theDeriv -= probVec[altId] * probVec[j] * scale * V[j] ;
	  }
	}
	if (isfinite(theDeriv) == 0) {
	  DEBUG_MESSAGE("Error in deriv mu") ;
	  (*success) = patFALSE ;
	  *muDerivative = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
	if (snpTerms) {
	  *muDerivative += theDeriv * factorForDerivOfSnpTerms ;
	}
	else {
	  *muDerivative += theDeriv ;
	}
      }
    
      // Derivative with respect to the scale parameter

      if (compScaleDerivative) {
	theDeriv = 0.0 ;
	theDeriv += (*mu) * probVec[altId] * V[altId] ;
	for ( j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    theDeriv -= (*mu) * probVec[altId] * probVec[j] * V[j] ;
	  }
	}
	if (isfinite(theDeriv) == 0) {
	  (*success) = patFALSE ;
	  (*scaleDerivative) = patMaxReal ;
	  ++overflow ;
	  return patMaxReal ;
	}
	if (snpTerms) {
	  (*scaleDerivative) += theDeriv * factorForDerivOfSnpTerms ;
	}
	else {
	  (*scaleDerivative) += theDeriv ;
	}
      }
    }
  }

  (*success) = patTRUE ;

  if (logOfProba) {
    result = scaleMuV[altId] - lsumExp ;
    if (!patFinite(result)) {
      ++underflow ;
      result = -patMaxReal ;
    }
    
    if (isfinite(result) == 0) {
      stringstream str ;
      str << "Cannot compute " << (*mu) << "*" <<  scale << "*" <<  V[altId] << "-log(" 
	  << sumExp << ")" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      (*success) = patFALSE ;
      return patReal() ;
    }
    // Second derivatives

    
    if (secondDeriv != NULL) {

      if (!utility->isLinear()) {
	err = new patErrMiscError("Second derivatives are computed only for linear in parameters models") ;
	WARNING(err->describe()) ;
	return patMaxReal ;
      }
            
      for (k = 0 ; k < nBeta ; ++k) {
	term[k] = 0.0 ;
	for (j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    term[k] += utilDerivatives[j][k] * probVec[j] ;
	  }
	}
      }
      for (k = 0 ; k < nBeta ; ++k) {
	for (kk = k ; kk < nBeta ; ++kk) {
	  theHessianEntry = 0.0 ;
	  for (j = 0 ; j < J ; ++j) {
	    if (individual->availability[j]) {
	      theHessianEntry += probVec[j] * 
		(utilDerivatives[j][k] - term[k]) *
		(utilDerivatives[j][kk] - term[kk]) ;
	    }
	  }
	  secondDeriv->secondDerivBetaBeta[kk][k] += theHessianEntry ; 
	  if (k != kk) {
	    secondDeriv->secondDerivBetaBeta[k][kk] += theHessianEntry ;
	  }
	}
      }
    }
    return result ;
  }
  else {
    return probVec[altId] ;
  }
}

patString patProbaMnlModel::getModelName(patError*& err) {
  return patString("Logit Model") ;
}


patReal patProbaMnlModel::getMinExpArgument() {
  return minExpArgument ;
}
patReal patProbaMnlModel::getMaxExpArgument() {
  return maxExpArgument ;
}

unsigned long patProbaMnlModel::getUnderflow() {
  return underflow ;
}

unsigned long patProbaMnlModel::getOverflow() {
  return overflow ;
}

patString patProbaMnlModel::getInfo() {
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


void patProbaMnlModel::generateCppCodePerDraw(ostream& cppFile,
					      patBoolean logOfProba,
					      patBoolean derivatives, 
					      patBoolean secondDerivatives, 
					      patError*& err) {

  patBoolean trueHessian = (patModelSpec::the()->isSimpleMnlModel() && patParameters::the()->getBTRExactHessian() && patModelSpec::the()->isMuFixed()) ;
  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;
  
  cppFile << "    //////////////////////////////////" << endl ;
  cppFile << "    // Code generated in patProbaMnlModel" << endl ;
  if (!patModelSpec::the()->groupScalesAreAllOne()) {
    cppFile << "  unsigned long scaleIndex = groupIndex[observation->group] ;" << endl ;
    cppFile << "  patReal defaultScale = scalesPerGroup[observation->group] ;" ;
  }
  utility->genericCppCode(cppFile,derivatives,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  cppFile << "  patReal sumExp = 0.0 ;" << endl ;
  cppFile << "  patReal minExpArgument(1.0) ;" << endl ;
  cppFile << "  patReal maxExpArgument(1.0) ;" << endl ;
  cppFile << "  for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
  cppFile << "    if (observation->availability[j]) {" << endl ;
  if (!patModelSpec::the()->groupScalesAreAllOne()) {
  
    cppFile << "  if (scaleIndex != " << patBadId << ") {" << endl ;
    cppFile << " // Scale parameter for the group" << endl ;
    cppFile << "  V[j] *= (*x)[scaleIndex] ;" << endl ;
    if (derivatives) {
      cppFile << "      for (unsigned long k = 0 ; k < grad->size()  ; ++k) {" << endl ;
      cppFile << "	  DV[k][j] *= (*x)[scaleIndex] ;" << endl ;
      cppFile << "      }" << endl ;
    }
    cppFile << "  }" << endl ;
    cppFile << "    else {" << endl ;
    cppFile << "      V[j] *= defaultScale ;" << endl ;
    cppFile << "    }" << endl ;
  }
  patBetaLikeParameter theMu = 
    patModelSpec::the()->getMu(err)  ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (!theMu.isFixed) {
    cppFile << " // Mu parameter" << endl ;
    cppFile << "  V[j] *= (*x)[" << theMu.index << "]  ;" << endl ;
  }
  else {
    if (theMu.defaultValue != 1.0) {
      cppFile << "  V[j] *= " << theMu.defaultValue << " ; " << endl ;
    }
  }
  cppFile << "  if (V[j] > maxExpArgument) {" << endl ;
  cppFile << "    maxExpArgument = V[j] ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  if (V[j] < minExpArgument) {" << endl ;
  cppFile << "    minExpArgument = V[j] ;" << endl ;
  cppFile << "  }" << endl ;

  cppFile << "  if (V[j] >= " << patLogMaxReal::the() << ") {" << endl ;
  cppFile << "    expV[j] = " << patMaxReal << " ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  else {" << endl ;
  cppFile << "    expV[j] = exp(V[j]) ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  sumExp += expV[j] ;" << endl ;
  cppFile << "  } //if (observation->availability[j]  " << endl ;
  cppFile << "  } //for ( j = 0 ; j < "<< J <<" ; ++j) " << endl ;
  cppFile << "  patReal lsumExp = log(sumExp) ;" << endl ;

  if (derivatives) {
    cppFile << "  for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
    cppFile << "    probVec[j] = expV[j] / sumExp ;" << endl ;
    cppFile << "  };" << endl ;
  }

  cppFile << "  patReal result = V[altIndex[observation->choice]] - lsumExp;" << endl ;
  cppFile << "  if (!patFinite(result)) {" << endl ;
  cppFile << "    result = -" << patMaxReal << " ;" << endl ;
  cppFile << "  }" << endl ;
  
  cppFile << "  if (isfinite(result) == 0) {" << endl ;
  cppFile << "    err = new patErrMiscError(\"Numerical problem when computing the model\") ;" << endl ;
  cppFile << "    WARNING(err->describe()) ;" << endl ;
  cppFile << "    (*success) = patFALSE ;" << endl ;
  cppFile << "    return NULL;" << endl ;
  cppFile << "  }" << endl ;
  
  if (logOfProba) {
    cppFile << "    logProbaForOnlyDraw = result ;" << endl ;
  }
  else {
    cppFile << "    probaPerDraw = exp(result) ;" << endl ;
  }
  if (derivatives) {
    stringstream computeMu ;
    computeMu << " " ;
    if (!theMu.isFixed) {
      computeMu << "(*x)[" << theMu.index << "] * " ;
    }
    else {
      if (theMu.defaultValue != 1.0) {
	computeMu << theMu.defaultValue << " * " ;
      }
    }

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
    cppFile << gradientName.str() << "[k] = "
	    << computeMu.str() 
	    << " DV[k][altIndex[observation->choice]] " 
	    << whenNoLog.str() << " ;" << endl ;
    cppFile << "	for ( unsigned long j = 0 ; j < " << J << " ; ++j) {" << endl ;
    cppFile << "	if (observation->availability[j]) {" << endl ;
    cppFile << gradientName.str() << "[k] -= " << computeMu.str() << " probVec[j] * DV[k][j] " << whenNoLog.str() <<";" << endl ;
    cppFile << "	} //if (observation->availability[j])" << endl ;
    cppFile << "	} // for ( j = 0 ; j < " << J << " ; ++j) " << endl ;
    cppFile << "  } // for (unsigned long k = 0 ; k < " << K <<" ; ++k)" << endl ;

    if (!theMu.isFixed) {
      cppFile << "    // Derivatives with respect to the mu parameter" << endl ;
      cppFile << "      " << gradientName.str() << "[" << theMu.index <<"] =   V[altIndex[observation->choice]]  / (*x)["<<theMu.index<<"];" << endl ;
      cppFile << "      for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
      cppFile << "	if (observation->availability[j]) {" << endl ;
      cppFile << "	  " << gradientName.str() << "[" << theMu.index <<"] -= probVec[j] * V[j]  / (*x)["<<theMu.index<<"];" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "      }" << endl ;
      
    }

    if (patModelSpec::the()->estimateGroupScales()) {
      cppFile << "    // Derivative with respect to the scale parameter" << endl ;
      cppFile << "  if (scaleIndex != patBadId) {" << endl ;
      cppFile << " // Scale parameter for the group" << endl ;
      cppFile << "   " << gradientName.str() << "[scaleIndex] = V[altIndex[observation->choice]] / (*x)[scaleIndex] ;"  << endl;
      cppFile << "      for (unsigned long j = 0 ; j < "<<J <<" ; ++j) {" << endl ;
      cppFile << "	if (observation->availability[j]) {" << endl ;
      cppFile << "	   " << gradientName.str() << "[scaleIndex] -= probVec[j] * V[j] / (*x)[scaleIndex];" << endl ;
      cppFile << "  }" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "      }" << endl ;
    }    
    
  }

  if (secondDerivatives) {
    if (!trueHessian) {
      return  ;
    }
    
    nBeta = patModelSpec::the()->getNbrNonFixedParameters() ;

    cppFile << "    vector<patReal> term(" << nBeta << ",0.0) ;" << endl ;
    cppFile << "    for (unsigned long k = 0 ; k < " << nBeta <<" ; ++k) {" << endl ;
    cppFile << "      for (unsigned long j = 0 ; j < " << J << " ; ++j) {" << endl ;
    cppFile << "	if (observation->availability[j]) {" << endl ;
    cppFile << "	  term[k] += DV[k][j] * probVec[j] ;" << endl ;
    cppFile << "	}" << endl ;
    cppFile << "      }" << endl ;
    cppFile << "    }" << endl ;

    if (trueHessian) {
      cppFile << "    for (unsigned long k = 0 ; k < " << nBeta << " ; ++k) {" << endl ;
      cppFile << "      for (unsigned long kk = k ; kk < " << nBeta << " ; ++kk) {" << endl ;
      cppFile << "	patReal theHessianEntry = 0.0 ;" << endl ;
      cppFile << "	for (unsigned long j = 0 ; j < " << J << " ; ++j) {" << endl ;
      cppFile << "	  if (observation->availability[j]) {" << endl ;
      cppFile << "	    theHessianEntry += probVec[j] * " << endl ;
      cppFile << "	      (DV[k][j] - term[k]) *" << endl ;
      cppFile << "	      (DV[kk][j] - term[kk]) ;" << endl ;
      cppFile << "	  }" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "	trueHessian->addElement(kk,k,theHessianEntry,err) ; " << endl ;
      cppFile << "      }" << endl ;
      cppFile << "    } // for (unsigned long k = 0 ; k < " << nBeta << " ; ++k)" << endl ;
    }
  }
  cppFile << "    // End of code generated in patProbaMnlModel" << endl ;
  cppFile << "    //////////////////////////////////" << endl ;

}
