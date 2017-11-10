//-*-c++-*------------------------------------------------------------
//
// File name : patProbaMnlPanelModel.cc
// Author :    Michel Bierlaire
// Date :      Tue Mar 30 14:18:53 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patUtility.h"
#include "patModelSpec.h"
#include "patProbaMnlPanelModel.h"
#include "patMath.h"
#include "patGEV.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patRandomDraws.h"
#include "patValueVariables.h"
#include "patTimer.h"

patProbaMnlPanelModel::patProbaMnlPanelModel(patUtility* aUtility) :
patProbaPanelModel(aUtility), minExpArgument(1.0),maxExpArgument(1.0),overflow(0),underflow(0)  {

}

patProbaMnlPanelModel::~patProbaMnlPanelModel() {
  DELETE_PTR(betaDrawDerivatives) ;
  DELETE_PTR(paramDrawDerivatives) ;
  DELETE_PTR(muDrawDerivative) ;
  DELETE_PTR(scaleDrawDerivative) ;

}


patReal patProbaMnlPanelModel::evalProbaPerObs(patObservationData* individual,
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
    *success = patFALSE ;
    return patReal() ;
  }

  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;
  }

  unsigned long J = patModelSpec::the()->getNbrAlternatives() ;

  if (utility == NULL) {
    err = new patErrNullPointer("patUtility") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;

  }

  if (!noDerivative) {
    if (betaDerivatives == NULL || paramDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal() ;
    }
    if (compMuDerivative && muDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal() ;
    }
    if (compScaleDerivative && scaleDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal() ;
    }
  }

  if (beta == NULL || parameters == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;
  }


  if (x == NULL) {
    err = new patErrNullPointer("vector<patVariables>") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  if (!noDerivative) {
    if (betaDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal()  ;
    }
    if (paramDerivatives == NULL) {
      err = new patErrNullPointer("patVariables") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal()  ;
    }
    if (muDerivative == NULL) {
      err = new patErrNullPointer("patReal") ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal()  ;
    }
  }

  if (mu == NULL) {
    err = new patErrNullPointer("patReal") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  // Check if the sizes and index are compatibles

  //  DEBUG_MESSAGE("--> debug 2") ;
  unsigned long altId = patModelSpec::the()->getAltInternalId(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  if (x->size() != patModelSpec::the()->getNbrUsedAttributes()) {
    err = new patErrMiscError("Incompatible nbr of characteristics") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  if (individual->availability.size() != J) {
    err = new patErrMiscError("Incompatible nbr of availabilities") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  if (altId >= J) {
    err = new patErrOutOfRange<unsigned long>(altId,0,J-1) ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  // Check if the chosen alternative is available

  if (!individual->availability[altId]) {
    stringstream str ;
    str << "Alternative " << altId << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  //    patTimer::the()->toc(0) ;
  // At this point, everything seems to be ready to compute
  //  DEBUG_MESSAGE("--> debug 3") ;

  //  For each available alternative compute j , V_j, e^{V_j}, et G_j 

  patVariables V(J) ;
  patVariables expV(J) ;

  patVariables scaleMuV(J) ;
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
	*success = patFALSE ;
	return patReal() ;
      }
      if (V[j] > maxUtility) {
	maxUtility = V[j] ;
      }
    }
  }
  
  //  patTimer::the()->toc(2) ;

  //  patTimer::the()->tic(3) ;

  patReal sumExp(0.0) ;

  for (unsigned long j = 0 ; j < J ; ++j) {
    if (individual->availability[j]) {
      //  DEBUG_MESSAGE("--> debug 4") ;
      V[j] -= maxUtility;
      scaleMuV[j] = scale*(*mu)*V[j] ;
      if (scaleMuV[j] > maxExpArgument) {
	maxExpArgument = scaleMuV[j] ;
      }
      if (scaleMuV[j] < minExpArgument) {
	minExpArgument = scaleMuV[j] ;
      }
      if (scaleMuV[j] >= patLogMaxReal::the()) {
	WARNING("Overflow for alt " << j << ": exp(" << (*mu)*scale*V[j] << ")") ;
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

  if (!patFinite(sumExp)) {
    sumExp = patMaxReal ;
  }
  //  patTimer::the()->toc(3) ;

  patVariables probVec(J) ;
  patReal lsumExp = log(sumExp) ;

  if (noDerivative) {

    patReal result ;
    if (!patFinite(scaleMuV[altId])) {
      ++underflow ;
      result = -patMaxReal ;
    }
    else {
      result = scaleMuV[altId] - lsumExp;
      if (!patFinite(result)) {
	++underflow ;
	result = -patMaxReal ;
      }
    }
    
    if (isfinite(result) == 0) {
      stringstream str ;
      str << "Cannot compute " << (*mu) << "*" <<  scale << "*" <<  V[altId] << "-log(" 
	  << sumExp << ")" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      (*success) = patFALSE ;
      *success = patFALSE ;
      return patReal() ;
    }
    

    patReal theProba = exp(result) ;
    *success = patTRUE ;
    return theProba ;
  }
  else {
    for (unsigned long j = 0 ; j < J ; ++j) {
      if (individual->availability[j]) {
	patReal tmp = scaleMuV[j] - lsumExp;
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
	  *success = patFALSE ;
	  return patReal() ;
	}
	probVec[j] = exp(tmp) ;
      }
    } 
  }

  if (!noDerivative) {
    // Derivatives with respect to beta

    unsigned long nBeta = compBetaDerivatives.size() ;

    if (nBeta != patModelSpec::the()->getNbrTotalBeta()) {
      stringstream str ;
      str << "There should be " << patModelSpec::the()->getNbrTotalBeta() 
	  << " beta's instead of " << nBeta ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal() ;
    }
    

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
	  *success = patFALSE ;
	  return patReal() ;
	}

	for (unsigned long k = 0 ; k < nBeta ; ++k) {
	  if (compBetaDerivatives[k]) {
	    DV[k][i] = scale * utilDerivatives[k] ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      *success = patFALSE ;
	      return patReal()  ;
	    }
	  }
	}
      }
    }
    //    patTimer::the()->toc(5) ;
    //    patTimer::the()->tic(6) ;

    for (unsigned long k = 0 ; k < nBeta ; ++k) {
      if (compBetaDerivatives[k]) {
	(*betaDerivatives)[k] += (*mu) * DV[k][altId] * probVec[altId] ;
	for (unsigned long j = 0 ; j < J ; ++j) {
	  if (individual->availability[j]) {
	    (*betaDerivatives)[k] -= 
	      (*mu) * probVec[altId] * probVec[j] * DV[k][j] ;
	  }
	}
      }
    }

    // Derivatives with respect to the mu parameter

    if (compMuDerivative) {

      (*muDerivative) += probVec[altId] * scale * V[altId] ;
      for (unsigned long j = 0 ; j < J ; ++j) {
	if (individual->availability[j]) {

	  (*muDerivative) -= 
	    probVec[altId] * probVec[j] * scale * V[j] ;
	}
      }
      if (isfinite(*muDerivative) == 0) {
	DEBUG_MESSAGE("Error in deriv mu") ;
	(*success) = patFALSE ;
	*muDerivative = patMaxReal ;
	++overflow ;
	*success = patFALSE ;
	return patMaxReal ;
      }
    }
    
    // Derivative with respect to the scale parameter

    if (compScaleDerivative) {


      (*scaleDerivative) += (*mu) * probVec[altId] * V[altId] ;
      for (unsigned long j = 0 ; j < J ; ++j) {
	if (individual->availability[j]) {
	  (*scaleDerivative) -= (*mu) * probVec[altId] * probVec[j] * V[j] ;
	}
      }
      if (isfinite((*scaleDerivative)) == 0) {
	(*success) = patFALSE ;
	(*scaleDerivative) = patMaxReal ;
	++overflow ;
	*success = patFALSE ;
	return patMaxReal ;
      }
    }
  }

  (*success) = patTRUE ;

  return probVec[altId] ;
}

patString patProbaMnlPanelModel::getModelName(patError*& err) {
  return patString("Logit Model for panel data") ;
}


patReal patProbaMnlPanelModel::getMinExpArgument() {
  return minExpArgument ;
}
patReal patProbaMnlPanelModel::getMaxExpArgument() {
  return maxExpArgument ;
}

unsigned long patProbaMnlPanelModel::getUnderflow() {
  return underflow ;
}

unsigned long patProbaMnlPanelModel::getOverflow() {
  return overflow ;
}

patString patProbaMnlPanelModel::getInfo() {
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


void patProbaMnlPanelModel::generateCppCodePerObs(ostream& cppFile,
						  patBoolean derivatives, 
						  patError*& err) {

  // The "gradientPerDraw" vector actually accumulates the gradients
  // of the logs.
  //

  unsigned K = patModelSpec::the()->getNbrNonFixedParameters() ;
  
  cppFile << "    //////////////////////////////////" << endl ;
  cppFile << "    // Code generated in patProbaMnlPanelModel" << endl ;
  unsigned long J = patModelSpec::the()->getNbrAlternatives() ; 
  if (derivatives) {
    cppFile << "    vector<vector<patReal> > DV(grad->size(),vector<patReal>("<< J <<",0.0)) ;" << endl ;
    cppFile << "    vector<patReal> probVec("<< J <<") ;" << endl ;
  }
  cppFile << "  patReal maxUtility = " << -patMaxReal <<" ;" << endl ;
  cppFile << "    vector<patReal> V("<<J<<") ;" << endl ;
  cppFile << "    vector<patReal> expV("<<J<<") ;" << endl ;
  cppFile << "  " << endl ;
  for (unsigned long j = 0 ; 
       j < J ;
       ++j) {
    cppFile << "    // Alternative " << j << endl ;
    cppFile << "    if (observation->availability[" << j << "]) {" << endl ;
      
    cppFile << "  V["<<j<<"] = " ;
    utility->generateCppCode(cppFile,j,err) ;
   
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    cppFile << " ;" << endl ;
    if (derivatives) {
      for (unsigned long i = 0 ; i < K ; ++i) {
	cppFile << "      DV[" << i <<" ][" << j << "] = " ;
	utility->generateCppDerivativeCode(cppFile,j,i,err) ;
	cppFile << ";" << endl ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
    }

    cppFile << "    if (V["<<j<<"] > maxUtility) {" << endl ;
    cppFile << "	maxUtility = V["<<j<<"] ;" << endl ;
    cppFile << "      }//   if (V["<<j<<"] > maxUtility)" << endl ;
    cppFile << "    } //if (individual->availability[" << j << "])"<< endl ;
  }

  cppFile << "  patReal sumExp = 0.0 ;" << endl ;
  cppFile << "  patReal minExpArgument(1.0) ;" << endl ;
  cppFile << "  patReal maxExpArgument(1.0) ;" << endl ;
  cppFile << "  for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
  cppFile << "    if (observation->availability[j]) {" << endl ;
  cppFile << "      V[j] -= maxUtility;" << endl ;
  
  if (patModelSpec::the()->estimateGroupScales()) {
    cppFile << "  if (scaleIndex != patBadId) {" << endl ;
    cppFile << " // Scale parameter for the group" << endl ;
    cppFile << "  V[j] *= (*x)[scaleIndex] ;" << endl ;
    if (derivatives) {
      cppFile << "      for (unsigned long k = 0 ; k < grad->size()  ; ++k) {" << endl ;
      cppFile << "	  DV[k][j] *= (*x)[scaleIndex] ;" << endl ;
      cppFile << "      }" << endl ;
    }
    cppFile << "  }" << endl ;
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
  
  cppFile << "    sumOfLogs += result ;" << endl ;
  
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
    cppFile << "	for (unsigned long k = 0 ; k < " << K << " ; ++k) {" << endl ;
    
    cppFile << "gradientPerDraw[k] += "
	    << computeMu.str() 
	    << " DV[k][altIndex[observation->choice]] " 
	    << whenNoLog.str() << " ;" << endl ;
    cppFile << "	for ( unsigned long j = 0 ; j < " << J << " ; ++j) {" << endl ;
    cppFile << "	if (observation->availability[j]) {" << endl ;
    cppFile << "gradientPerDraw[k] -= " << computeMu.str() << " probVec[j] * DV[k][j] " << whenNoLog.str() <<";" << endl ;
    cppFile << "	} //if (observation->availability[j])" << endl ;
    cppFile << "	} // for ( j = 0 ; j < " << J << " ; ++j) " << endl ;
    cppFile << "  } // for (unsigned long k = 0 ; k < " << K <<" ; ++k)" << endl ;

    if (!theMu.isFixed) {
      cppFile << "    // Derivatives with respect to the mu parameter" << endl ;
      cppFile << "     gradientPerDraw[" << theMu.index <<"] +=   V[altIndex[observation->choice]]  / (*x)["<<theMu.index<<"];" << endl ;
      cppFile << "      for (unsigned long j = 0 ; j < "<< J <<" ; ++j) {" << endl ;
      cppFile << "	if (observation->availability[j]) {" << endl ;
      cppFile << "     gradientPerDraw[" << theMu.index <<"] -= probVec[j] * V[j]  / (*x)["<<theMu.index<<"];" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "      }" << endl ;
      
    }

    if (patModelSpec::the()->estimateGroupScales()) {
      cppFile << "    // Derivative with respect to the scale parameter" << endl ;
      cppFile << "  if (scaleIndex != patBadId) {" << endl ;
      cppFile << " // Scale parameter for the group" << endl ;
      cppFile << "  gradientPerDraw[scaleIndex] += V[altIndex[observation->choice]] / (*x)[scaleIndex] ;"  << endl;
      cppFile << "      for (unsigned long j = 0 ; j < "<<J <<" ; ++j) {" << endl ;
      cppFile << "	if (observation->availability[j]) {" << endl ;
      cppFile << "	  gradientPerDraw[scaleIndex] -= probVec[j] * V[j] / (*x)[scaleIndex];" << endl ;
      cppFile << "  }" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "      }" << endl ;
    }    
    
  }

  cppFile << "    // End of code generated in patProbaMnlPanelModel" << endl ;
  cppFile << "    //////////////////////////////////" << endl ;
}
