//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevAlt.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Dec 12 12:52:15 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patMath.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patNetworkGevAlt.h"
#include "patNetworkGevIterator.h"

patNetworkGevAlt::patNetworkGevAlt(const patString& name,
				   unsigned long muIndex,
				   unsigned long _altIndex) : 
  nodeName(name),
  indexOfMu(muIndex),
  altIndex(_altIndex) {
  relevant.insert(_altIndex) ;

  //  DEBUG_MESSAGE("Create node " << name << " with index " << _altIndex) ;
}

patReal patNetworkGevAlt::evaluate(const patVariables* x,
				   const patVariables* param,
				   const patReal* mu,
				   const vector<patBoolean>& available,
				   patError*& err) {
  
  //  DEBUG_MESSAGE("Evaluate alt" << getModelName()) ;
  if (!available[altIndex]) {
    return 0.0 ;
  }
  
  patReal res =  pow((*x)[altIndex],(*param)[indexOfMu]) ;
  return res ;
}

patReal patNetworkGevAlt::getDerivative_xi(unsigned long index,
					   const patVariables* x,
					   const patVariables* param,
					   const patReal* mu,
					   const vector<patBoolean>& available,
					   patError*& err) {
  
  patReal finDiff = getDerivative_xi_finDiff(index,x,param,mu,available,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }


  if (!available[altIndex]) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  if (index != altIndex) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  patReal mui = (*param)[indexOfMu] ;
  patReal res = (mui * pow((*x)[index],mui-1.0)) ;
  if (patAbs(finDiff-res) > 1.0e-6) {
    DEBUG_MESSAGE("Err = " << patAbs(finDiff-res) << " Deriv: " << res << " Finite Diff: " <<finDiff) ;
  }
  return res ;

}

patReal patNetworkGevAlt::getDerivative_mu(const patVariables* x,
					const patVariables* param,
					const patReal* mu,
					const vector<patBoolean>& available,
					patError*& err) {
  return 0.0 ;
}

patReal patNetworkGevAlt::getDerivative_param(unsigned long index,
					      const patVariables* x,
					      const patVariables* param,
					      const patReal* mu, 
					      const vector<patBoolean>& available,
					      patError*& err) {
  patReal finDiff =  getDerivative_param_finDiff(index,
						 x,
						 param,
						 mu,
						 available,
						 err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  
  if (!available[altIndex]) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  if (index != indexOfMu) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  patReal mui = (*param)[indexOfMu] ;
  patReal xval = (*x)[altIndex] ;
  patReal res = (log(xval) * pow(xval,mui)) ;
  if (patAbs(finDiff-res) > 1.0e-6) {
    DEBUG_MESSAGE("Err = " << patAbs(finDiff-res) << " Deriv: " 
		  << res << " Finite Diff: " <<finDiff) ;
  }
  return res ;
  
}

patReal patNetworkGevAlt::getSecondDerivative_xi_xj(unsigned long index1,
						    unsigned long index2,
						    const patVariables* x,
						    const patVariables* param,
						    const patReal* mu,
						    const vector<patBoolean>& available,
						    patError*& err) {
  
  patReal finDiff = getSecondDerivative_xi_xj_finDiff( index1,
						       index2,
						       x,
						       param,
						       mu,
						       available,
						       err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  //  return finDiff ;
  
  if (!available[altIndex]) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  if (index1 != index2 || index1 != altIndex) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  patReal mui = (*param)[indexOfMu] ;
  patReal xval = (*x)[altIndex] ;
  patReal res = (mui * (mui-1.0) * pow(xval,mui - 2.0)) ;
  if (patAbs(finDiff-res) > 1.0e-6) {
    DEBUG_MESSAGE("Err = " << patAbs(finDiff-res) << " Deriv: " << res << " Finite Diff: " <<finDiff) ;
  }
  return res ;
}

patReal patNetworkGevAlt::getSecondDerivative_xi_mu(unsigned long index,
						    const patVariables* x,
						    const patVariables* param,
						    const patReal* mu,
						    const vector<patBoolean>& available,
						     patError*& err) {
  return 0.0 ;
}

patReal patNetworkGevAlt::getSecondDerivative_param(unsigned long indexVar,
						     unsigned long indexParam,
						     const patVariables* x,
						     const patVariables* param,
						     const patReal* mu, 
						     const vector<patBoolean>& available,

						     patError*& err) {
  patReal finDiff = getSecondDerivative_param_finDiff(indexVar,
						      indexParam,
						      x,
						      param,
						      mu,
						      available,
						      err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }


  if (!available[altIndex]) {
    return 0.0 ;
  }
  if (!available[indexVar]) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  if (indexParam != indexOfMu) {
    if (patAbs(finDiff)> patEPSILON) {
      DEBUG_MESSAGE("Err = " << patAbs(finDiff) << " Deriv: 0.0 Finite Diff: " <<finDiff) ;
    }
    return 0.0 ;
  }
  patReal mui = (*param)[indexOfMu] ;
  patReal xval = (*x)[indexVar] ;
  patReal res = (pow(xval,mui-1.0) + mui * log(xval) * pow(xval,mui-1.0)) ;
  if (patAbs(finDiff-res) > 1.0e-6) {
    DEBUG_MESSAGE("Err = " << patAbs(finDiff-res) << " Deriv: " << res << " Finite Diff: " <<finDiff) ;
  }
  return res ;
  
}
   
patString patNetworkGevAlt::getModelName() {
  return patString("Alt " + nodeName) ;
}

unsigned long patNetworkGevAlt::getNbrParameters() {
  WARNING("**** Should not be called");
  return (-1) ;
}

unsigned long patNetworkGevAlt::nSucc() {
  return 0 ;
}

patBoolean patNetworkGevAlt::isRoot() {
  return patFALSE ;
}


patString patNetworkGevAlt::nodeType() const  {
  return patString("Alt") ;
}

unsigned long patNetworkGevAlt::getMuIndex() {
  return indexOfMu ;
}

ostream& patNetworkGevAlt::print(ostream& str) {
  str << "Alt " << nodeName << endl ;
  str << "   Index of mu= " << indexOfMu << endl ;
  return str ;

}

void patNetworkGevAlt::addSuccessor(patNetworkGevNode* aNode,
				    unsigned long index) {
  WARNING("Cannot add node " << aNode->getModelName() << " to an alternative") ;
}

patIterator<patNetworkGevNode*>* patNetworkGevAlt::getSuccessorsIterator() {
  patIterator<patNetworkGevNode*>* res = new patNetworkGevIterator(NULL) ;
  return res ;
}

patBoolean patNetworkGevAlt::isAlternative() {
  return patTRUE ;
}

patString patNetworkGevAlt::getNodeName() {
  return nodeName ;
}

set<unsigned long> patNetworkGevAlt::getRelevantAlternatives() {
  return relevant ;
}
