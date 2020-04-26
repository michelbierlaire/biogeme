//-*-c++-*------------------------------------------------------------
//
// File name : patCNL.cc
// Author :    Michel Bierlaire
// Date :      Fri Aug 11 00:33:09 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <fstream>
#include "patMath.h"
#include "patPower.h"
#include "patCNL.h"
#include "patModelSpec.h"
#include "patParameters.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
 
patCNL::
patCNL() : firstTime(patTRUE) {
  
}

patCNL::~patCNL() {
  for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
    DELETE_PTR(alphaIter[nest]) ;
  }
}


// Evaluate the function  
patReal patCNL::evaluate(const patVariables* x,
			 const patVariables* param,
			 const patReal* mu,
			 const vector<patBoolean>& available,
			 patError*& err) {


  err = new patErrMiscError("This function should not be called") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

  // Compute the partial derivatives with respect to the variables

patReal patCNL::getDerivative_xi(unsigned long index,
				 const patVariables* x,
				 const patVariables* param,
				 const patReal* mu,
				 const vector<patBoolean>& available,
				 patError*& err) {

  if (!available[index]) {
    stringstream str ;
    str << "Alt. " << index << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  //  DEBUG_MESSAGE("First deriv[" << index << "]= " << firstDeriv_xi[index]) ;
  return firstDeriv_xi[index] ;

}

  // Compute the partial derivative with respect to mu
patReal patCNL::getDerivative_mu(const patVariables* x,
				 const patVariables* param,
				 const patReal* mu,
				 const vector<patBoolean>& available,
				 patError*& err) {
   
  err = new patErrMiscError("Function getDerivative_mu should not be called") ;
  WARNING(err->describe()) ;
  return patReal()  ;
}


// Compute the partial derivatives with respect to the parameters
patReal patCNL::getDerivative_param(unsigned long index,
				       const patVariables* x,
				       const patVariables* param,
				       const patReal* mu, 
				       const vector<patBoolean>& available,
				       patError*& err) {
  
  err = new patErrMiscError("Function getDerivative_param should not be called") ;
  WARNING(err->describe()) ;
  return patReal()  ;
}

patReal patCNL::getSecondDerivative_xi_xj(unsigned long index1,
					  unsigned long index2,
					  const patVariables* x,
					  const patVariables* param,
					  const patReal* mu,
					  const vector<patBoolean>& available,
					  patError*& err) {

  if (!available[index1]) {
    stringstream str ;
    str << "Alt. " << index1 << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!available[index2]) {
    stringstream str ;
    str << "Alt. " << index2 << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  //  DEBUG_MESSAGE("secondDeriv[" << index1 << "][" << index2 <<"]= " << secondDeriv_xi_xj[index1][index2]) ;
  return secondDeriv_xi_xj[index1][index2] ;
  
}

patString patCNL::getModelName() { 
  return patString("Cross-Nested Logit Model") ;
}

unsigned long patCNL::getNbrParameters() {
  return (patModelSpec::the()->getNbrModelParameters()) ;
}


patReal patCNL::getSecondDerivative_xi_mu(unsigned long index,
					  const patVariables* x,
					  const patVariables* param,
					  const patReal* mu,
					  const vector<patBoolean>& available,
					  patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  //    DEBUG_MESSAGE("muDeriv[" << index << "]= " << muDerivative[index]) ;
  return muDerivative[index] ;
}


patReal patCNL::getSecondDerivative_param(unsigned long indexVar,
					  unsigned long indexParam,
					  const patVariables* x,
					  const patVariables* param,
					  const patReal* mu, 
					  const vector<patBoolean>& available,
					  patError*& err) {

  //    DEBUG_MESSAGE("secondXiPram[" << indexVar << "][" << indexParam << "]=" << secondDeriv_xi_param[indexVar][indexParam]) ;
  return secondDeriv_xi_param[indexVar][indexParam] ;
}

   



void  patCNL::compute(const patVariables* x,
		      const patVariables* param,
		      const patReal* mu, 
		      const vector<patBoolean>& available,
		      patBoolean computeSecondDerivatives,
		      patError*& err) {

  //  DEBUG_MESSAGE("COMPUTE GEV DERIVATIVES") ;

  nNests = patModelSpec::the()->getNbrNests() ;
  J = patModelSpec::the()->getNbrAlternatives() ;
  if (firstTime) {

    indexOfNestParam.resize(nNests,patBadId) ;
    indexOfAlphaParam.resize(nNests,vector<unsigned long>(J,patBadId)) ;
    nestForParamIndex.resize(param->size(),patBadId) ;
    altForParamIndex.resize(param->size(),patBadId) ;
    nestParams.resize(nNests,1.0) ;
    alphas.resize(nNests,vector<patReal>(J,0.0)) ;
    alphasToMumOverMu.resize(nNests,vector<patReal>(J,0.0)) ;
    xToMum.resize(nNests,vector<patReal>(J,0.0)) ;
    alphaIter.resize(nNests,NULL) ;
    Am.resize(nNests,0.0) ;
    Bm.resize(nNests,0.0) ;
    Delta.resize(J,0.0) ;
    DAmDMum.resize(nNests,0.0) ;
    DAmDMu.resize(nNests,0.0) ;
    firstDeriv_xi.resize(J,0.0) ;
    secondDeriv_xi_xj.resize(J,vector<patReal>(J,0.0)) ;
    secondDeriv_xi_param.resize(J,vector<patReal>(param->size(),0.0)) ;
    muDerivative.resize(J,0.0) ;
    computeDerivativeParam.resize(param->size(),patFALSE) ;

    for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
      indexOfNestParam[nest] = patModelSpec::the()->getIdCnlNestCoef(nest) ;
      alphaIter[nest] = 
	patModelSpec::the()->getAltIteratorForNonZeroAlphas(nest) ;
      if (alphaIter[nest] == NULL) {
	err = new patErrNullPointer("patIterator<unsigned long>") ;
	WARNING(err->describe()); 
	return ;
      }
    }

    for (unsigned long paramIndex = 0 ; 
	 paramIndex < param->size() ; 
	 ++paramIndex) {
      pair<unsigned long,unsigned long> nestAlt = 
	patModelSpec::the()->getNestAltFromParamIndex(paramIndex) ;
      nestForParamIndex[paramIndex] = nestAlt.first ;
      altForParamIndex[paramIndex] = nestAlt.second ;
    }
    patIterator<patBetaLikeParameter>* iter =
      patModelSpec::the()->createAllModelIterator()  ;
    for (iter->first() ;
	 !iter->isDone() ;
	 iter->next()) {
      patBetaLikeParameter aParam = iter->currentItem() ;
      computeDerivativeParam[aParam.id] = !aParam.isFixed ;
    }

    for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
      for (alphaIter[nest]->first() ;
	   !alphaIter[nest]->isDone() ;
	   alphaIter[nest]->next()) {
	unsigned long alt = alphaIter[nest]->currentItem() ;
	unsigned long paramIndex = 
	  patModelSpec::the()->getIdCnlAlpha(nest,alt) ;
	indexOfAlphaParam[nest][alt] = paramIndex ;
      }
    }

    patBetaLikeParameter muParam = patModelSpec::the()->getMu(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    
    computeMuDerivative = !muParam.isFixed ;
    firstTime = patFALSE ;
  } 

  fill(firstDeriv_xi.begin(),firstDeriv_xi.end(),0.0) ;
  fill(secondDeriv_xi_xj.begin(),secondDeriv_xi_xj.end(),vector<patReal>(J,0.0)) ;
  fill(secondDeriv_xi_param.begin(),secondDeriv_xi_param.end(),vector<patReal>(param->size(),0.0)) ;
  fill(muDerivative.begin(),muDerivative.end(),0.0) ;


  for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
    nestParams[nest] = (*param)[indexOfNestParam[nest]] ;
    patReal mumOverMu = nestParams[nest] / (*mu) ;
    Am[nest] = 0.0 ;

    for (alphaIter[nest]->first() ;
	 !alphaIter[nest]->isDone() ;
	 alphaIter[nest]->next()) {
      unsigned long alt = alphaIter[nest]->currentItem() ;
      alphas[nest][alt] = (*param)[indexOfAlphaParam[nest][alt]] ;
      alphasToMumOverMu[nest][alt] = 
	patPower(alphas[nest][alt],mumOverMu) ;
      xToMum[nest][alt] = patPower((*x)[alt],nestParams[nest]) ;
      if (available[alt]) {
	Am[nest] += alphasToMumOverMu[nest][alt] * xToMum[nest][alt] ;
      }
    }
    Bm[nest] = patPower(Am[nest],(*mu)/nestParams[nest]) ;
  }

  
  // First derivatives
  
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    if (available[alt]) {
      Delta[alt] = 0.0 ;
      for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
	if (indexOfAlphaParam[nest][alt] != patBadId) {
	  if (Bm[nest] != 0 && alphasToMumOverMu[nest][alt] != 0 && xToMum[nest][alt] != 0) {
	    Delta[alt] += 
	      Bm[nest] * alphasToMumOverMu[nest][alt] * xToMum[nest][alt] / 
	      (Am[nest] * (*x)[alt]) ;
	  }
	}
      }

      firstDeriv_xi[alt] = (*mu) * Delta[alt] ;

      if (computeSecondDerivatives) {
	// Second derivatives
	for (unsigned long altJ = 0 ; altJ < J ; ++altJ) {
	  if (available[altJ]) {
	    for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
	      if (indexOfAlphaParam[nest][altJ] != patBadId) {
		if (((*mu)-nestParams[nest]) != 0 &&  
		    Bm[nest] != 0 &&
		    xToMum[nest][alt] != 0 && 
		    xToMum[nest][altJ] != 0 &&
		    alphasToMumOverMu[nest][alt] != 0 &&
		    alphasToMumOverMu[nest][altJ] != 0) {
		  secondDeriv_xi_xj[alt][altJ] += 
		    ((*mu)-nestParams[nest]) * 
		    Bm[nest] * 
		    xToMum[nest][alt] * 
		    xToMum[nest][altJ] * 
		    alphasToMumOverMu[nest][alt] * 
		    alphasToMumOverMu[nest][altJ] / 
		    (Am[nest] * Am[nest] * (*x)[alt] * (*x)[altJ]) ;
		}
		if (alt == altJ) {
		  if (Bm[nest] != 0 &&
		      alphasToMumOverMu[nest][alt] != 0 &&
		      (nestParams[nest] - 1.0) != 0 && 
		      xToMum[nest][altJ] != 0) {
		    secondDeriv_xi_xj[alt][altJ] +=
		      Bm[nest] * 
		      alphasToMumOverMu[nest][alt] * 
		      (nestParams[nest] - 1.0) * 
		      xToMum[nest][altJ] / 
		      (Am[nest] * (*x)[alt] * (*x)[alt]) ;
		  }
		}
	      }
	    }
	    secondDeriv_xi_xj[alt][altJ] *= (*mu) ;
	  }
	}
      }
    }
  }

  if (!computeSecondDerivatives) {
    return ;
  }

  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    if (available[alt]) {

      muDerivative[alt] = 0.0 ;

      // With respect to mu

      if (computeMuDerivative) {
	
	for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
	  if (indexOfAlphaParam[nest][alt] != patBadId) {

	    patReal derivAm(0.0) ;
	    
	    for (unsigned long j = 0 ; j < J ; ++j) {
	      if ((indexOfAlphaParam[nest][j] != patBadId) && available[j]) {
		if (alphasToMumOverMu[nest][j] != 0 && 
		    xToMum[nest][j] != 0 &&
		    nestParams[nest] != 0) {
		  derivAm -= alphasToMumOverMu[nest][j] * 
		    xToMum[nest][j] * 
		    log(alphas[nest][j]) * nestParams[nest] ;
		}
	      }
	    }
	    derivAm /= (*mu) * (*mu) ;

	    patReal fm(0) ;
	    if (Bm[nest] != 0) {
	      fm = Bm[nest] / Am[nest] ;
	    }

	    patReal derivFm(0) ;
	    if (fm != 0) {
	      patReal term(0) ;
	      if ((*mu)/nestParams[nest] - 1.0 != 0 && derivAm != 0) {
		term = ((*mu)/nestParams[nest] - 1.0) * derivAm / Am[nest] ;
	      }
	      term +=  log(Am[nest]) / nestParams[nest] ;
	      derivFm = fm * term ;
	    }
	    
	    patReal derivGm(0) ; 
	    if (alphasToMumOverMu[nest][alt] != 0 &&
		nestParams[nest] != 0) {
	      derivGm = -alphasToMumOverMu[nest][alt] *
		log(alphas[nest][alt]) * 
		nestParams[nest] / ((*mu) * (*mu)) ;
	    }
	    
	    if (derivFm != 0 &&
		alphasToMumOverMu[nest][alt] != 0 &&
		xToMum[nest][alt] != 0) {
	      muDerivative[alt] +=
		derivFm * 
		alphasToMumOverMu[nest][alt] * 
		xToMum[nest][alt] / 
		(*x)[alt] ;
	    }

	    if (fm != 0 &&
		derivGm != 0 &&  
		xToMum[nest][alt] != 0) {
	      muDerivative[alt] += fm *
		derivGm *  
		xToMum[nest][alt] / 
		(*x)[alt] ;
	    }

	  }
	}
	muDerivative[alt] *= (*mu) ;
	muDerivative[alt] += Delta[alt] ;
	//	DEBUG_MESSAGE("muDerivativeFinal["<<  alt << "]=" << muDerivative[alt]) ;
	
      }
      // With respect to mum
      
      for (unsigned long nest = 0 ; nest < nNests ; ++nest) {
	unsigned long nestParamIndex = indexOfNestParam[nest] ;
	secondDeriv_xi_param[alt][nestParamIndex] = 0.0 ;
	if (indexOfAlphaParam[nest][alt] != patBadId) {
	  if (computeDerivativeParam[nestParamIndex]) {
	    patReal fm(0) ;
	    if (Bm[nest] != 0) {
	      fm = Bm[nest] / Am[nest] ;
	    }
	    patReal hm = xToMum[nest][alt] / (*x)[alt] ;
	    
	    // Derivative of Am w.r.t mum
	    
	    patReal derivAm(0.0) ;
	    for (unsigned long j = 0 ; j < J ; ++j) {	  
	      if ((indexOfAlphaParam[nest][j] != patBadId) && available[j]) {
		if (alphasToMumOverMu[nest][j] != 0 &&
		    xToMum[nest][j] != 0) {
		  derivAm += 
		    alphasToMumOverMu[nest][j] * 
		    xToMum[nest][j] * 
		    (log((*x)[j]) + log(alphas[nest][j]) / (*mu) ) ;
		}
	      }
	    }
	    
	    // Derivative of fm w.r.t. mu

	    patReal derivFm(0) ;
	    if (fm != 0) {
	      patReal term(0)  ;
	      if (((*mu)/nestParams[nest] - 1.0) != 0 && derivAm != 0) {
		term = ((*mu)/nestParams[nest] - 1.0) * derivAm / Am[nest] ;
	      }
	      term -= (*mu) * log(Am[nest]) / (nestParams[nest] * nestParams[nest]) ;
	      
	      derivFm = fm * term ;
	    }
	    
	    secondDeriv_xi_param[alt][nestParamIndex] = 
	      derivFm * alphasToMumOverMu[nest][alt] * hm  ;
	    if (fm != 0.0 &&  hm != 0.0 && alphasToMumOverMu[nest][alt] != 0.0) {
	      secondDeriv_xi_param[alt][nestParamIndex] += 
		fm * hm * alphasToMumOverMu[nest][alt] * log(alphas[nest][alt]) / (*mu) ;
	    }
	    
	    if (fm != 0 && alphasToMumOverMu[nest][alt] != 0 && xToMum[nest][alt] !=0) {
	      secondDeriv_xi_param[alt][nestParamIndex] += 
		fm * 
		alphasToMumOverMu[nest][alt] * 
		xToMum[nest][alt] * 
		log((*x)[alt]) / (*x)[alt] ;
	    }
	    
	    //	    DEBUG_MESSAGE("XXX=" << secondDeriv_xi_param[alt][nestParamIndex]) ;

	    secondDeriv_xi_param[alt][nestParamIndex] *= (*mu) ;
	  }
	  
	
	  // With respect to alpha's
	  
	  for (unsigned long j = 0 ; j < J ; ++j) {
	    if (available[j]) {
	      unsigned long alphaParamIndex = indexOfAlphaParam[nest][j] ;
	      if (alphaParamIndex != patBadId) {
		secondDeriv_xi_param[alt][alphaParamIndex] = 0.0 ;
		if (computeDerivativeParam[alphaParamIndex]) {
		  patReal derivAm(0) ;
		  if ((nestParams[nest] / (*mu)) != 0 &&
		      alphasToMumOverMu[nest][j] != 0 &&
		      xToMum[nest][j] != 0) {
		    derivAm = (nestParams[nest] / (*mu)) * 
		      alphasToMumOverMu[nest][j] * 
		      xToMum[nest][j] / alphas[nest][j] ;
		  }
		  if ((*mu) != 0 &&
		      (((*mu)/nestParams[nest]) - 1.0) != 0 && 
		      Bm[nest] != 0 && 
		      derivAm != 0 && 
		      alphasToMumOverMu[nest][alt] != 0 &&
		      xToMum[nest][alt] != 0) {

		    secondDeriv_xi_param[alt][alphaParamIndex] =
		      (*mu) * (((*mu)/nestParams[nest]) - 1.0) * 
		      Bm[nest] * 
		      derivAm * 
		      alphasToMumOverMu[nest][alt] * 
		      xToMum[nest][alt] / 
		      (Am[nest] * Am[nest] * (*x)[alt]) ;
		  }
		  else {
		    secondDeriv_xi_param[alt][alphaParamIndex] = 0.0 ;
		  }
		  
		  if (alt == j) {
		    if (nestParams[nest] != 0 && 
			Bm[nest] != 0 && 
			alphasToMumOverMu[nest][alt] != 0 &&
			xToMum[nest][alt]!= 0) {
		      secondDeriv_xi_param[alt][alphaParamIndex] +=
			nestParams[nest] * 
			Bm[nest] * 
			alphasToMumOverMu[nest][alt] * 
			xToMum[nest][alt]  / 
			(Am[nest] * alphas[nest][alt] * (*x)[alt]) ;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
}


void patCNL::generateCppCode(ostream& cppFile, 
			     patBoolean derivatives, 
			     patError*& err) {
  err = new patErrMiscError("CNL: not yet implemented") ;
  WARNING(err->describe()) ;
  return ;
}
