//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevNest.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Dec 10 18:31:32 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <algorithm>
#include "patMath.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patNetworkGevNest.h"
#include "patNetworkGevIterator.h"

patNetworkGevNest::patNetworkGevNest(const patString& name,
				     unsigned long ind) : 
  nodeName(name),
  indexOfMu(ind),
  relevantComputed(patFALSE) {

}

patReal patNetworkGevNest::evaluate(const patVariables* x,
				const patVariables* param,
				const patReal* mu,
				const vector<patBoolean>& available,
				patError*& err) {
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }

  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal res = 0.0 ;
  patReal mui ;
  if (isRoot()) {
    mui = *mu ;
  }
  else {
    mui = (*param)[indexOfMu] ;
  }
  for (unsigned long i = 0 ; i < nSucc() ; ++i) {
    patNetworkGevNode* theModel = listOfSuccessors[i] ;
    patReal alpha = (*param)[indexOfAlpha[i]] ;
    patReal muj = (*param)[theModel->getMuIndex()] ;
    patReal Gj = theModel->evaluate(x,
				    param,
				    mu,
				    available,
				    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal term = (Gj == 0) ? 0.0 : alpha * pow(Gj,mui/muj) ;
    if (!isfinite(term)) {
      WARNING("Numerical problem: " 
	      << alpha 
	      << "*pow(" 
	      << Gj << "," 
	      << mui << "/" << muj << ")") ;
    }
    res +=  term ;
  }
  return res ;
}

patReal patNetworkGevNest::getDerivative_xi(unsigned long index,
					const patVariables* x,
					const patVariables* param,
					const patReal* mu,
					const vector<patBoolean>& available,
					patError*& err) {

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (!available[index]) {
    return 0.0 ;
  }

  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal res = 0.0 ;
  patReal mui ;
  if (isRoot()) {
    mui = *mu ;
  }
  else {
    mui = (*param)[indexOfMu] ;
  }
  for (unsigned long i = 0 ; i < nSucc() ; ++i) {
    patNetworkGevNode* theModel = listOfSuccessors[i] ;
    patReal alpha = (*param)[indexOfAlpha[i]] ;
    patReal muj = (*param)[theModel->getMuIndex()] ;
    patReal ratio = mui/muj ;
    patReal Gj = theModel->evaluate(x,
				    param,
				    mu,
				    available,
				    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal dGjxk = theModel->getDerivative_xi(index,
					       x,
					       param,
					       mu,
					       available,
					       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    if (dGjxk != 0.0) {
      patReal term = (Gj == 0.0) 
	? 0.0 
	: alpha * ratio * pow(Gj,ratio-1.0) * dGjxk ;
      if (!isfinite(term)) {
	WARNING("Numerical problem: " << alpha << "*" << ratio << "*pow(" << Gj << "," << ratio-1.0 << ")*" << dGjxk) ;
      }
      res +=  term ;
    }
  }
  return res ;
}

patReal patNetworkGevNest::getDerivative_mu(const patVariables* x,
					const patVariables* param,
					const patReal* mu,
					const vector<patBoolean>& available,
					patError*& err) {


    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }

  if (!isRoot()) {
    return 0.0 ;
  }
  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal res = 0.0 ;
  patReal mui = *mu ;
  for (unsigned long i = 0 ; i < nSucc() ; ++i) {
    patNetworkGevNode* theModel = listOfSuccessors[i] ;
    patReal alpha = (*param)[indexOfAlpha[i]] ;
    patReal muj = (*param)[theModel->getMuIndex()] ;
    patReal ratio = mui/muj ;
    patReal Gj = theModel->evaluate(x,
				    param,
				    mu,
				    available,
				    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal term = (Gj == 0.0) 
      ? 0.0 
      : (alpha / muj) * pow(Gj,ratio) * log(Gj) ;
    res +=  term ;
  }
  return res ;

  
}

patReal patNetworkGevNest::getDerivative_param(unsigned long index,
					   const patVariables* x,
					   const patVariables* param,
					   const patReal* mu, 
					   const vector<patBoolean>& available,
					   patError*& err) {

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }

  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal mui ;
  if (isRoot()) {
    mui = *mu ;
  }
  else {
    mui = (*param)[indexOfMu] ;
  }
  // Check if the param is the mu associated with the node
  if (index == indexOfMu) {
    patReal res = 0.0 ;
    for (unsigned long i = 0 ; i < nSucc() ; ++i) {
      patNetworkGevNode* theModel = listOfSuccessors[i] ;
      patReal alpha = (*param)[indexOfAlpha[i]] ;
      patReal muj = (*param)[theModel->getMuIndex()] ;
      patReal ratio = mui/muj ;
      patReal Gj = theModel->evaluate(x,
				      param,
				      mu,
				      available,
				      err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      patReal term = (Gj == 0.0) 
	? 0.0 
	: (alpha / muj) * pow(Gj,ratio) * log(Gj) ;
      res +=  term ;
    }

    return res ;

  }
  else {
    // Check if the parameter is one of the mu's of the successors
    // In the loop, check also if it is an alpha parameter.
    patNetworkGevNode* theSuccessor = NULL ;
    patReal alpha ;
    unsigned long alphaSucc = patBadId ;
    for (unsigned long i = 0 ; i < nSucc() ; ++i) {
      if (listOfSuccessors[i]->getMuIndex() == index) {
	theSuccessor = listOfSuccessors[i] ;
	alpha = (*param)[indexOfAlpha[i]] ;
	break ;
      }
      if (indexOfAlpha[i] == index) {
	alphaSucc = i ;
	break ;
      }
    }
    if (theSuccessor != NULL) {
      // This is a mu parameter
      patReal Gj = theSuccessor->evaluate(x,
					  param,
					  mu,
					  available,
					  err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      patReal dGjmuj = theSuccessor->getDerivative_param(index,
							 x,
							 param,
							 mu, 
							 available,
							 err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      patReal muj = (*param)[index] ;
      patReal ratio = mui / muj ;
      patReal ratio2 = mui / (muj * muj) ;
      patReal res = (Gj == 0.0) 
	? 0.0 
	: (alpha * ratio) * pow(Gj,ratio-1.0) * dGjmuj
	- alpha * ratio2 * pow(Gj,ratio) * log(Gj) ;
      return res ;
    }
    else {
      if (alphaSucc == patBadId) {
	// This is not an alpha parameter. 
	patReal res = 0.0 ;
	for (unsigned long i = 0 ; i < nSucc() ; ++i) {	  
	  patReal dG_param = 
	    listOfSuccessors[i]->getDerivative_param(index,
						     x,
						     param,
						     mu, 
						     available,
						     err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }
	  if (dG_param != 0.0) {
	    patReal Gj = listOfSuccessors[i]->evaluate(x,
						       param,
						       mu,
						       available,
						       err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return patReal() ;
	    }
	    patReal muj = (*param)[listOfSuccessors[i]->getMuIndex()] ;
	    patReal alpha = (*param)[indexOfAlpha[i]] ;
	    patReal ratio = mui / muj ;
	    patReal term = (Gj == 0.0) 
	      ? 0.0 
	      : alpha * ratio * dG_param * pow(Gj,ratio-1.0) ;
	    res +=  term ;
	  }
	}
	return res ;
      }
      else {
	// This is an alpha parameter
	patNetworkGevNode* theModel = listOfSuccessors[alphaSucc] ;
	patReal muj = (*param)[theModel->getMuIndex()] ;
	patReal Gj = theModel->evaluate(x,
					param,
					mu,
					available,
					err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	patReal res = (Gj == 0.0) ? 0.0 : pow(Gj,mui/muj) ;
	return res ;
      }
    }
  }
}

patReal patNetworkGevNest::getSecondDerivative_xi_xj(unsigned long index1,
						 unsigned long index2,
						 const patVariables* x,
						 const patVariables* param,
						 const patReal* mu,
						 const vector<patBoolean>& available,
						 patError*& err) {

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  if (!available[index1] || !available[index2]) {
    return 0.0 ;
  }

  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal res = 0.0 ;
  patReal mui ;
  if (isRoot()) {
    mui = *mu ;
  }
  else {
    mui = (*param)[indexOfMu] ;
  }
  for (unsigned long i = 0 ; i < nSucc() ; ++i) {
    patNetworkGevNode* theModel = listOfSuccessors[i] ;
    patReal alpha = (*param)[indexOfAlpha[i]] ;
    patReal muj = (*param)[theModel->getMuIndex()] ;
    patReal ratio = mui/muj ;
    patReal Gj = theModel->evaluate(x,
				    param,
				    mu,
				    available,
				    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal dGjxk = theModel->getDerivative_xi(index1,
					       x,
					       param,
					       mu,
					       available,
					       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal dGjxm = theModel->getDerivative_xi(index2,
					       x,
					       param,
					       mu,
					       available,
					       err) ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal d2Gjxkxm = theModel->getSecondDerivative_xi_xj(index1,
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

    patReal term(0.0) ;
    if (Gj != 0.0) {
      if (dGjxk != 0.0 && dGjxm != 0.0) {
	term += alpha * ratio * (ratio - 1.0) * pow(Gj,ratio-2.0) * dGjxk * dGjxm ;
      }
      if (d2Gjxkxm != 0.0) {
	term += alpha * ratio * pow(Gj,ratio-1.0) * d2Gjxkxm ;
      }      
      res +=  term ;
    }
  }

  return res ;
}

patReal patNetworkGevNest::getSecondDerivative_xi_mu(unsigned long index,
						     const patVariables* x,
						     const patVariables* param,
						     const patReal* mu,
						     const vector<patBoolean>& available,
						     patError*& err) {

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  if (!available[index]) {
    return 0.0 ;
  }

  if (!isRoot()) {
    return 0.0 ;
  }
  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }
  patReal res = 0.0 ;
  patReal mui = *mu ;
  for (unsigned long i = 0 ; i < nSucc() ; ++i) {
    patNetworkGevNode* theModel = listOfSuccessors[i] ;
    patReal alpha = (*param)[indexOfAlpha[i]] ;
    patReal muj = (*param)[theModel->getMuIndex()] ;
    patReal ratio = mui/muj ;
    patReal Gj = theModel->evaluate(x,
				    param,
				    mu,
				    available,
				    err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal dGjxk = theModel->getDerivative_xi(index,
					       x,
					       param,
					       mu,
					       available,
					       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal term = (Gj == 0.0) 
      ? 0.0 
      : (alpha / muj) * pow(Gj,ratio-1.0) * dGjxk 
      + (ratio * alpha / muj) * pow(Gj,ratio - 1.0)  * log(Gj) * dGjxk ;
    res += term ;
  }

	
  return res ;

}

patReal patNetworkGevNest::getSecondDerivative_param(unsigned long indexVar,
						     unsigned long indexParam,
						     const patVariables* x,
						     const patVariables* param,
						     const patReal* mu, 
						     const vector<patBoolean>& available,
						     patError*& err) {

    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  if (!available[indexVar]) {
    return 0.0 ;
  }

  if (nSucc() == 0) {
    stringstream str ;
    str << "No successor has been defined for " << getModelName() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return patReal() ;
  }

  patReal mui;
  if (isRoot()) {
    mui = *mu ;
  }
  else {
    mui = (*param)[indexOfMu] ;
  }
  
  // Check if the param is the mu associated with the node
  if (indexParam == indexOfMu) {
    patReal res = 0.0 ;
    for (unsigned long i = 0 ; i < nSucc() ; ++i) {
      patNetworkGevNode* theModel = listOfSuccessors[i] ;
      patReal alpha = (*param)[indexOfAlpha[i]] ;
      patReal muj = (*param)[theModel->getMuIndex()] ;
      patReal ratio = mui/muj ;
      patReal Gj = theModel->evaluate(x,
				      param,
				      mu,
				      available,
				      err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      
      patReal dGjxk = theModel->getDerivative_xi(indexVar,
						 x,
						 param,
						 mu,
						 available,
						 err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }


      patReal term = (Gj == 0.0) 
	? 0.0 
	: (alpha / muj)  * pow(Gj,ratio -1.0) * dGjxk
	+ (alpha * ratio / muj) * pow(Gj,ratio - 1.0) * log(Gj) * dGjxk ;
      res += term ;
    }
    return res ;
  }
  else {
    // Check if the parameter is one of the mu's of the successors
    // In the loop, check also if it is an alpha parameter.
    patNetworkGevNode* theSuccessor = NULL ;
    patReal alpha ;
    unsigned long alphaSucc = patBadId ;
    for (unsigned long i = 0 ; i < nSucc() ; ++i) {
      if (listOfSuccessors[i]->getMuIndex() == indexParam) {
	theSuccessor = listOfSuccessors[i] ;
	alpha = (*param)[indexOfAlpha[i]] ;
	break ;
      }
      if (indexOfAlpha[i] == indexParam) {
	alphaSucc = i ;
	break ;
      }
    }
    if (theSuccessor != NULL) {
      // This is a mu parameter
      patReal Gj = theSuccessor->evaluate(x,
					  param,
					  mu,
					  available,
					  err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      patReal dGjmuj = theSuccessor->getDerivative_param(indexParam,
							 x,
							 param,
							 mu, 
							 available,
							 err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      patReal dGjxk = theSuccessor->getDerivative_xi(indexVar,
						     x,
						     param,
						     mu,
						     available,
						     err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patReal() ;
      }
      patReal d2Gimujxk = 
	theSuccessor->getSecondDerivative_param( indexVar,
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
      patReal muj = (*param)[indexParam] ;
      patReal ratio = mui / muj ;
      patReal ratio2 = mui / (muj * muj) ;
      patReal res = (Gj == 0.0) ? 0.0 :
	alpha * ratio * (ratio - 1.0) * pow(Gj,ratio-2.0) * dGjmuj * dGjxk
	+ alpha * ratio * pow(Gj,ratio-1.0)*d2Gimujxk
	- alpha * ratio2 * ratio * pow(Gj,ratio-1.0) * dGjxk * log(Gj) 
	- alpha * ratio2 * pow(Gj,ratio-1) * dGjxk ;

      return res ;
    }
    else {
      if (alphaSucc == patBadId) {
      // This is not an alpha parameter

	patReal res = 0.0 ;
	for (unsigned long i = 0 ; i < nSucc() ; ++i) {
	  patNetworkGevNode* theModel = listOfSuccessors[i] ;
	  patReal alpha = (*param)[indexOfAlpha[i]] ;
	  patReal muj = (*param)[theModel->getMuIndex()] ;
	  patReal ratio = mui/muj ;
	  patReal Gj = theModel->evaluate(x,
					  param,
					  mu,
					  available,
					  err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }
	  patReal dGjxk = theModel->getDerivative_xi(indexVar,
						     x,
						     param,
						     mu,
						     available,
						     err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }
	  patReal dGjgamma = theModel->getDerivative_param(indexParam,
							   x,
							   param,
							   mu,
							   available,
							   err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return patReal() ;
	  }
	  patReal d2Gjxkgamma = theModel->getSecondDerivative_param(indexVar,
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
	  
	  patReal term = (Gj == 0.0) 
	    ? 0.0 
	    : alpha * ratio * (ratio - 1.0) * pow(Gj,ratio-2.0) 
	    * dGjxk * dGjgamma 
	    + alpha * ratio * pow(Gj,ratio-1.0) * d2Gjxkgamma ;
	  res += term ;
	}
	return res ;
      }
      else {
	// This is an alpha parameter
	patNetworkGevNode* theModel = listOfSuccessors[alphaSucc] ;
	patReal muj = (*param)[theModel->getMuIndex()] ;
	patReal ratio = mui / muj ;
	patReal Gj = theModel->evaluate(x,
					param,
					mu,
					available,
					err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	patReal dGjxk = theModel->getDerivative_xi(indexVar,
						   x,
						   param,
						   mu,
						   available,
						   err) ;

	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	

	patReal res = (Gj == 0.0) ? 0.0 : ratio * pow(Gj,ratio-1.0) * dGjxk ;
	return res ;
      }
    }
  }
}

patString patNetworkGevNest::getModelName() {
  return patString("Network GEV Node " + nodeName) ;
}

unsigned long patNetworkGevNest::getNbrParameters() {
  if (isRoot()) {
    return patNetworkGevNode::getNbrParameters() ;
  }
  WARNING("*** This should not be called") ;
  return 0 ;
}

void patNetworkGevNest::addSuccessor(patNetworkGevNode* aNode, 
				     unsigned long index) {
  listOfSuccessors.push_back(aNode) ;
  indexOfAlpha.push_back(index) ;
}

unsigned long patNetworkGevNest::nSucc() {
  return listOfSuccessors.size() ;
}

unsigned long patNetworkGevNest::getMuIndex() {
  return indexOfMu ;
}

patBoolean patNetworkGevNest::isRoot() {
  return (indexOfMu == patBadId) ;
}

ostream& patNetworkGevNest::print(ostream& str) {
  if (!relevantComputed) {
    getRelevantAlternatives() ;
  }
  str << "Nest " << nodeName  ;
  str << " - Relevant alt:" ;
  for (set<unsigned long>::iterator i = relevant.begin() ;
       i != relevant.end() ;
       ++i) {
    str << *i << " " ;
  }
  str << endl ;
  str << "   Index of mu= " << indexOfMu << endl ;
  for (unsigned long i = 0 ;
       i < listOfSuccessors.size() ;
       ++i) {
    str << "-----" << listOfSuccessors[i]->getModelName() << "  alpha index=" << indexOfAlpha[i] << endl ;
  }
  for (unsigned long i = 0 ;
       i < listOfSuccessors.size() ;
       ++i) {
    listOfSuccessors[i]->print(str) ;
    str << endl ;
  }
  return str ;
}

patString patNetworkGevNest::nodeType() const {
  return patString("Nest") ;
}

patIterator<patNetworkGevNode*>* patNetworkGevNest::getSuccessorsIterator() {
  patIterator<patNetworkGevNode*>* res = 
    new patNetworkGevIterator(&listOfSuccessors) ;
  return res ;  
}

patBoolean patNetworkGevNest::isAlternative() {
  return patFALSE ;
}

patString patNetworkGevNest::getNodeName() {
  return nodeName ;
}

set<unsigned long> patNetworkGevNest::getRelevantAlternatives() {
  if (!relevantComputed) {
    //    DEBUG_MESSAGE("Compute relevant alt for node " << nodeName) ;
    for (vector<patNetworkGevNode*>::iterator aNode = listOfSuccessors.begin() ;
	 aNode != listOfSuccessors.end() ;
	 ++aNode) {
      std::set<unsigned long> childRelevant = (*aNode)->getRelevantAlternatives() ;
      for (std::set<unsigned long>::iterator i = childRelevant.begin() ;
	   i != childRelevant.end() ;
	   ++i) {
	//	DEBUG_MESSAGE("Insert " << *i) ;
	relevant.insert(*i) ;
      }
    }
    relevantComputed = patTRUE ;
  }
  return relevant ;
}
