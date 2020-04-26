//-*-c++-*------------------------------------------------------------
//
// File name : patNL.cc
// Author :    Michel Bierlaire
// Date :      Wed Sep  6 09:02:15 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include <algorithm>
#include "patNL.h"
#include "patPower.h"
#include "patModelSpec.h"
#include "patParameters.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"

patNL::patNL(patError*& err) : nestNames(patModelSpec::the()->getNbrNests()),
			       altPerNest(patModelSpec::the()->getNbrNests()), 
			       nestOfAlt(patModelSpec::the()->getNbrAlternatives()),
			       firstTime(patTRUE) {
  

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (patModelSpec::the()->getNbrNests() == 0) {
    err = new patErrMiscError("At least one nest must be defined for Nested Logit Model") ;
    WARNING(err->describe()) ;
    return ;
  }

  readNestRepartition(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return   ;
  }
}

patNL::~patNL() {

}

// Evaluate the function  
patReal patNL::evaluate(const patVariables* x,
			    const patVariables* param,
			    const patReal* mu,
			    const vector<patBoolean>& available,
			    patError*& err) {


  err = new patErrMiscError("This function should not be called") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

  // Compute the partial derivatives with respect to the variables

patReal patNL::getDerivative_xi(unsigned long index,
				const patVariables* x,
				const patVariables* param,
				const patReal* mu,
				const vector<patBoolean>& available,
				patError*& err) {



  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  if (!available[index]) {
    stringstream str ;
    str << "Alt. " << index << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  return firstDeriv_xi[index] ;

}

  // Compute the partial derivative with respect to mu
patReal patNL::getDerivative_mu(const patVariables* x,
				const patVariables* param,
				const patReal* mu,
				const vector<patBoolean>& available,
				patError*& err) {
  
  err = new patErrMiscError("Function getDerivative_mu should not be called") ;
  WARNING(err->describe()) ;
  return patReal()  ;
  
}


// Compute the partial derivatives with respect to the parameters
patReal patNL::getDerivative_param(unsigned long index,
				   const patVariables* x,
				   const patVariables* param,
				   const patReal* mu, 
				   const vector<patBoolean>& available,
				   patError*& err) {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  err = new patErrMiscError("Function getDerivative_param should not be called") ;
  WARNING(err->describe()) ;
  return patReal()  ;
}

  
void patNL::readNestRepartition(patError*& err) {

  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }

  patModelSpec::the()->assignNLNests(this,err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  
}
void patNL::addNestName(unsigned long nestId, 
			const patString& name,
			patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }

  if (nestId >= nestNames.size()) {
    err = new patErrOutOfRange<unsigned long>(nestId,
					     0,
					     nestNames.size()) ;
    WARNING(err->describe()) ;
    return  ;
  }
  nestNames[nestId] = name ;
}

void patNL::assignAltToNest(unsigned long altid,
			    const patString& name,
			    patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  unsigned long index = patModelSpec::the()->getAltInternalId(altid,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  vector<patString>::iterator iter = find(nestNames.begin(),
					  nestNames.end(),
					  name) ;

  if (iter == nestNames.end()) {
    stringstream str ;
    str << "Unknow nest " << name ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return  ;
  }

  unsigned long nestId = iter-nestNames.begin() ;
  
  altPerNest[nestId].push_back(index) ;
  nestOfAlt[index] = nestId ;


}
 
ostream& operator<<(ostream& str, const patNL& nl) {
  
  str << "Nested Logit model" << endl ;
  str << "Nbr of alternatives: " 
      << patModelSpec::the()->getNbrAlternatives() << endl ;
  str << "Nbr of nests: " << patModelSpec::the()->getNbrNests() << endl ;
  for (unsigned long i = 0 ; i < nl.nestNames.size() ; ++i) {
    str << "  " << nl.nestNames[i] << ":" ;
    for (list<unsigned long>::const_iterator iter = nl.altPerNest[i].begin() ;
	 iter != nl.altPerNest[i].end() ;
	 ++iter) {
      str << *iter << " " ;
    }
    str << endl ;
  }
  return str ;
}


patReal patNL::getSecondDerivative_xi_xj(unsigned long index1,
					 unsigned long index2,
					 const patVariables* x,
					 const patVariables* param,
					 const patReal* mu,
					 const vector<patBoolean>& available,
					 patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  if (!available[index1]) {
    stringstream str ;
    str << "Alt. " << index1 << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!available[index2]) {
    stringstream str ;
    str << "Alt. " << index2 << " is not available"  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  return secondDeriv_xi_xj[index1][index2] ;

}


patReal patNL::getSecondDerivative_xi_mu(unsigned long index,
					 const patVariables* x,
					 const patVariables* param,
					 const patReal* mu,
					 const vector<patBoolean>& available,
					 patError*& err) {


    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal()  ;
    }
  
    return muDerivative[index] ;

}

patReal patNL::getSecondDerivative_param(unsigned long indexVar,
				  unsigned long indexParam,
				  const patVariables* x,
				  const patVariables* param,
				  const patReal* mu, 
				  const vector<patBoolean>& available,
				  patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal()  ;
  }
  
  return secondDeriv_xi_param[indexVar][indexParam] ;
}
   




unsigned long patNL::getNbrParameters() {

  return patModelSpec::the()->getNbrNests() ;
}

void patNL::compute(const patVariables* x,
		    const patVariables* param,
		    const patReal* mu, 
		    const vector<patBoolean>& available,
		    patBoolean computeSecondDerivatives,
		    patError*& err) {


  if (firstTime) {

    nNests = patModelSpec::the()->getNbrNests() ;
    J = patModelSpec::the()->getNbrAlternatives() ;
    xToMum.resize(J,0.0) ;
    Am.resize(nNests,0.0) ;
    Bm.resize(nNests,0.0) ;
    firstDeriv_xi.resize(J,0.0) ;
    secondDeriv_xi_xj.resize(J,vector<patReal>(J,0.0)) ;
    secondDeriv_xi_param.resize(J,vector<patReal>(param->size(),0.0)) ;
    muDerivative.resize(J,0.0) ;
    computeDerivativeParam.resize(param->size(),patFALSE) ;

    patIterator<patBetaLikeParameter>* iter =
      patModelSpec::the()->createAllModelIterator()  ;
    for (iter->first() ;
	 !iter->isDone() ;
	 iter->next()) {
      patBetaLikeParameter aParam = iter->currentItem() ;
      computeDerivativeParam[aParam.id] = !aParam.isFixed ;
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
    //    patReal mumOverMu = (*param)[nest] / (*mu) ;
    Am[nest] = 0.0 ;
    for (list<unsigned long>::iterator altPtr = altPerNest[nest].begin() ;
	 altPtr != altPerNest[nest].end() ;
	 ++altPtr) {
      if (available[*altPtr]) {
	xToMum[*altPtr] = patPower((*x)[*altPtr],(*param)[nest]) ;
	Am[nest] += xToMum[*altPtr] ;
      }
    }
    Bm[nest] = patPower(Am[nest],(*mu)/(*param)[nest]) ;
  }

  
  // First derivatives
  
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    if (available[alt]) {
      unsigned long theNest = nestOfAlt[alt] ;
      
      patReal num = (*mu) * Bm[theNest] * xToMum[alt] ;
      patReal denom = (Am[theNest] * (*x)[alt]) ;
      if (num == 0) {
	firstDeriv_xi[alt] = 0 ;
      }
      else {
	firstDeriv_xi[alt] = num / denom ;
      }


      if (!isfinite(firstDeriv_xi[alt])) {
	WARNING((*mu) << "*" << Bm[theNest] << "* "<< xToMum[alt] << "/ "<< 
		"("<<Am[theNest] <<"*"<< (*x)[alt]<<")") ;
      }

      if (computeSecondDerivatives) {
	// Second derivatives
	
	for (list<unsigned long>::iterator altPtr = altPerNest[theNest].begin() ;
	     altPtr != altPerNest[theNest].end() ;
	     ++altPtr) {
	  if (available[*altPtr]) {
	    
	    secondDeriv_xi_xj[alt][*altPtr] = 
	      ((*mu) - (*param)[theNest]) * 
	      Bm[theNest] * 
	      xToMum[alt] * 
	      xToMum[*altPtr] / 
	      (Am[theNest] * Am[theNest]  * (*x)[alt] * (*x)[*altPtr]) ;
	    
	    if (alt == *altPtr) {
	      secondDeriv_xi_xj[alt][alt] +=
		((*param)[theNest] - 1.0) * 
		Bm[theNest] *
		xToMum[alt] / (Am[theNest] * (*x)[alt] * (*x)[alt]) ;
	    }
	    secondDeriv_xi_xj[alt][*altPtr] *= (*mu) ;
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
      unsigned long theNest = nestOfAlt[alt] ;
      
      // With respect to mu
      
      if (computeMuDerivative) {
	
	muDerivative[alt] = (1.0 + 
			     (*mu) * log(Am[theNest]) 
			     / (*param)[theNest]) 
	  * Bm[theNest] * xToMum[alt] / (Am[theNest] * (*x)[alt]) ;
	

      }
      
      if (computeDerivativeParam[theNest]) {
	// With respect to mum
	
	// Derivative of Am w.r.t mum
	
	patReal derivAm(0.0) ;
	for (list<unsigned long>::iterator altPtr = 
	       altPerNest[theNest].begin() ;
	     altPtr != altPerNest[theNest].end() ;
	     ++altPtr) {
	  if (available[*altPtr]) {
	    derivAm += xToMum[*altPtr] * log((*x)[*altPtr]) ;
	  }
	}
	patReal mum = (*param)[theNest] ;
	patReal tt = ((((*mu)/mum) - 1.0) * derivAm / Am[theNest]) - 
	  ((*mu)/(mum*mum)) * log(Am[theNest]) + 
	  log((*x)[alt]) ;
	secondDeriv_xi_param[alt][theNest] = 
	  (*mu) * 
	  Bm[theNest] * 
	  xToMum[alt] * 
	  tt / 
	  (Am[theNest] * (*x)[alt]) ;
      }
    }
  }
}


void patNL::generateCppCode(ostream& cppFile, 
			    patBoolean derivatives, 
			    patError*& err) {
  
  cppFile << "  ///////////////////////////////////////" << endl ;
  cppFile << "  // Code generated in patNL" << endl ;
  unsigned long J = patModelSpec::the()->getNbrAlternatives() ;
  //  unsigned long K = patModelSpec::the()->getNbrNonFixedParameters() ;
  unsigned long nNests = patModelSpec::the()->getNbrNests() ;
  unsigned long nParam = patModelSpec::the()->getNbrModelParameters() ;
  cppFile << "  vector<patReal> xToMum(" << J << ",0.0) ;" << endl ;
  cppFile << "  vector<patReal> Am(" << nNests << ",0.0) ;" << endl ;
  cppFile << "  vector<patReal> Bm(" << nNests << ",0.0) ;" << endl ;
  cppFile << "  vector<patReal> firstDeriv_xi(" << J << ",0.0) ;" << endl ;
  if (derivatives) {
    cppFile << "    vector< vector<patReal> > secondDeriv_xi_xj(" << J << ",vector<patReal>(" << J << ",0.0)) ;" << endl ;
    cppFile << "    vector< vector<patReal> > secondDeriv_xi_param(" << J << ",vector<patReal>(" << nParam << ",0.0)) ;" << endl ;
    cppFile << "    vector<patReal> muDerivative(" << J << ",0.0) ;" << endl ;
    cppFile << "patReal derivAm ;" << endl ;
  }

  vector <patBoolean> computeDerivativeParam(nParam,patFALSE) ;
  patIterator<patBetaLikeParameter>* iter =
    patModelSpec::the()->createAllModelIterator()  ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    patBetaLikeParameter aParam = iter->currentItem() ;
    DEBUG_MESSAGE(aParam) ;
    computeDerivativeParam[aParam.id] = !aParam.isFixed ;
  }
  patBetaLikeParameter muParam = patModelSpec::the()->getMu(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patBoolean computeMuDerivative = !muParam.isFixed ;
  patBoolean muIsOne = (muParam.isFixed) && (muParam.defaultValue == 1.0) ;
  stringstream strMu ;
  if (muParam.isFixed) {
    strMu << muParam.defaultValue ;
  }
  else {
    strMu << "(*x)[" << muParam.index << "]" ;
  }

  vector<stringstream*> strMum(nestNames.size()) ;
  for (vector<patString>::iterator name = nestNames.begin() ; 
       name != nestNames.end() ; 
       ++name) {
    patBoolean found ;
    patBetaLikeParameter nestParam = patModelSpec::the()->getNlNest(*name,
								    &found) ;
    if (!found) {
      stringstream str ;
      str << "Nest " << *name << " is unknown" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    //    patBoolean mumIsOne = (nestParam.isFixed) && (nestParam.defaultValue == 1.0) ;
    strMum[nestParam.id] = new stringstream ;
    if (nestParam.isFixed) {
      (*strMum[nestParam.id]) << nestParam.defaultValue ;
    }
    else {
      (*strMum[nestParam.id]) << "(*x)[" << nestParam.index << "]" ;
    }
    
    cppFile << "    Am[" << nestParam.id << "] = 0.0 ;" << endl ;
    for (list<unsigned long>::iterator altPtr = altPerNest[nestParam.id].begin() ;
	 altPtr != altPerNest[nestParam.id].end() ;
	 ++altPtr) {
      cppFile << "      if (observation->availability[" << *altPtr << "]) {" << endl ;
      cppFile << "	xToMum[" << *altPtr << "] = patPower(expV[" << *altPtr << "],"<< strMum[nestParam.id]->str() <<") ;" << endl ;
      cppFile << "	Am[" << nestParam.id << "] += xToMum[" << *altPtr << "] ;" << endl ;
      cppFile << "      }" << endl ;
    }
    cppFile << "    Bm[" << nestParam.id << "] = patPower(Am[" << nestParam.id << "],"<< strMu.str() <<"/"<<strMum[nestParam.id]->str()<<") ;" << endl ;
    
  }

  // First derivatives
  
  for (unsigned long alt = 0 ; alt < J ; ++alt) {
    cppFile << "    if (observation->availability[" << alt << "]) {" << endl ;
    unsigned long theNest = nestOfAlt[alt] ;
    
    cppFile << "    patReal num = " << strMu.str() 
	    <<" * Bm[" << theNest << "] * xToMum[" << alt <<"] ;" << endl ;
    cppFile << "    patReal denom = (Am[" << theNest <<"] * expV[" << alt << "]) ;" << endl ;
    cppFile << "    if (num == 0) {" << endl ;
    cppFile << "      firstDeriv_xi[" << alt << "] = 0 ;" << endl ;
    cppFile << "    }" << endl ;
    cppFile << "    else {" << endl ;
    cppFile << "      firstDeriv_xi[" << alt << "] = num / denom ;" << endl ;
    cppFile << "    }" << endl ;
    
    if (derivatives) {
      // Second derivatives
      
      for (list<unsigned long>::iterator altPtr = altPerNest[theNest].begin() ;
	   altPtr != altPerNest[theNest].end() ;
	   ++altPtr) {
	cppFile << "	if (observation->availability[" << *altPtr << "]) {" << endl ;
	
	cppFile << "	  secondDeriv_xi_xj[" << alt << "][" << *altPtr << "] = " << endl ;
	cppFile << "	    (" << strMu.str() << " - " << strMum[theNest]->str() << ") * " << endl ;
	cppFile << "	    Bm[" << theNest << "] * " << endl ;
	cppFile << "	    xToMum[" << alt << "] * " << endl ;
	cppFile << "	    xToMum[" << *altPtr << "] / " << endl ;
	cppFile << "	    (Am[" << theNest << "] * Am[" << theNest << "]  * expV[" << alt << "] * expV[" << *altPtr << "]) ;" << endl ;
	
	if (alt == *altPtr) {
	  cppFile << "	  secondDeriv_xi_xj[" << alt << "][" << alt << "] +=" << endl ;
	  cppFile << "	    (" << strMum[theNest]->str() << " - 1.0) * " << endl ;
	  cppFile << "	    Bm[" << theNest << "] *" << endl ;
	  cppFile << "	    xToMum[" << alt << "] / (Am[" << theNest << "] * expV[" << alt << "] * expV[" << alt << "]) ;" << endl ;
	}
	if (!muIsOne) {
	  cppFile << "	  secondDeriv_xi_xj[" << alt << "][" << *altPtr << "] *= " << strMu.str() << " ;" << endl ;
	}
	cppFile << "	} // if (observation->availability[" << *altPtr << "])" << endl ;
      }
    }
    cppFile << "    } // if (observation->availability[" << alt << "]) " << endl ;
  }

  if (derivatives) {
    for (unsigned long alt = 0 ; alt < J ; ++alt) {
      cppFile << "    if (observation->availability[" << alt << "]) {" << endl ;
      unsigned long theNest = nestOfAlt[alt] ;
      
      // With respect to mu
      
      if (computeMuDerivative) {
	
	cppFile << "	muDerivative[" << alt << "] = (1.0 + " << endl ;
	cppFile << "			     strMu.str() * log(Am[" << theNest <<"]) " << endl ;
	cppFile << "			     / " << strMum[theNest]->str() << ") " << endl ;
	cppFile << "	  * Bm[" << theNest << "] * xToMum[" << alt << "] / (Am[" << theNest << "] * expV[" << alt << "]) ;" << endl ;
	
      
      }
      
      if (computeDerivativeParam[theNest]) {
	// With respect to mum
	
	// Derivative of Am w.r.t mum
	
	cppFile << "	derivAm = 0.0 ;" << endl ;
	for (list<unsigned long>::iterator altPtr = 
	       altPerNest[theNest].begin() ;
	     altPtr != altPerNest[theNest].end() ;
	     ++altPtr) {
	  cppFile << "	  if (observation->availability[" << *altPtr << "]) {" << endl ;
	  cppFile << "	    derivAm += xToMum[" << *altPtr << "] * log(expV[" << *altPtr << "]) ;" << endl ;
	    cppFile << "	  } // if (observation->availability[" << *altPtr << "]) {" << endl ;
	}
	cppFile << "	patReal tt = ((("<<strMu.str()<<"/" << strMum[theNest]->str()<< ") - 1.0) * derivAm / Am[" << theNest << "]) - " << endl ;
	cppFile << "	  ("<<strMu.str()<<"/(" << strMum[theNest]->str()<< "*" << strMum[theNest]->str()<< ")) * log(Am[" << theNest << "]) + " << endl ;
	cppFile << "	  log(expV[" << alt << "]) ;" << endl ;
	cppFile << "	secondDeriv_xi_param[" << alt << "][" << theNest << "] = " << endl ;
	cppFile << "	  "<<strMu.str()<<" * " << endl ;
	cppFile << "	  Bm[" << theNest << "] * " << endl ;
	cppFile << "	  xToMum[" << alt << "] * " << endl ;
	cppFile << "	  tt / " << endl ;
	cppFile << "	  (Am[" << theNest << "] * expV[" << alt << "]) ;" << endl ;
      }
      cppFile << "    } // if (observation->availability[" << alt << "])" << endl ;
    }
  
  }
  cppFile << "  // End of code generated in patNL" << endl ;
  cppFile << "  ///////////////////////////////////////" << endl ;

  for (vector<stringstream*>::iterator i = strMum.begin() ;
       i != strMum.end() ;
       ++i) {
    DELETE_PTR(*i) ;
  }
}
