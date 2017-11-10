//-*-c++-*------------------------------------------------------------
//
// File name : bioBayesianResults.cc
// Author :    Michel Bierlaire
// Date :      Thu Aug  2 08:32:47 2012
//
//--------------------------------------------------------------------

#include "bioBayesianResults.h"

bioBayesianResults::bioBayesianResults(): theBayesianDraws(NULL),varCovar(NULL) {

}

bioBayesianResults::bioBayesianResults(const bioBayesianResults& b) {
  theBayesianDraws = b.theBayesianDraws ;
  paramNames = b.paramNames ;
  if (b.varCovar != NULL) {
    varCovar = new patHybridMatrix(*(b.varCovar)) ;
  }
  else {
    varCovar = NULL ;
  }
  mean = b.mean ;
}

const bioBayesianResults& bioBayesianResults::operator=( const bioBayesianResults& b ) {
  theBayesianDraws = b.theBayesianDraws ;
  paramNames = b.paramNames ;
  if (varCovar != NULL) {
    DELETE_PTR(varCovar) ;
  }
  if (b.varCovar != NULL) {
    varCovar = new patHybridMatrix(*(b.varCovar)) ;
  }
  else {
    varCovar = NULL ;
  }
  mean = b.mean ;

  return b ;
}


bioBayesianResults::bioBayesianResults(vector<vector<patReal> >* d,
				       vector<patString> b):
  theBayesianDraws(d),
  paramNames(b),
  varCovar(NULL) {
}

bioBayesianResults::~bioBayesianResults() {
  if (varCovar != NULL) {
    DELETE_PTR(varCovar) ;
  }
}

patULong bioBayesianResults::nDraws() const {
  if (theBayesianDraws == NULL) {
    return 0 ;
  }
  return theBayesianDraws->size() ;
}

patULong bioBayesianResults::nBetas() const {
  if (theBayesianDraws == NULL) {
    return 0 ;
  }
  if (theBayesianDraws->empty()) {
    return 0 ;
  }
  return (*theBayesianDraws)[0].size() ;
}


void bioBayesianResults::computeStatistics(patError*& err) {
  if (paramNames.size() != nBetas()) {
    stringstream str ;
    str << "Incompatible sizes: " << paramNames.size() << " and " << nBetas() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  mean.erase(mean.begin(),mean.end()) ;

  patReal theMean ;
  for (patULong b = 0 ; b < nBetas() ; ++b) {
    theMean = 0.0 ;
    for (patULong d = 0 ; d < nDraws() ; ++d) {
      theMean += (*theBayesianDraws)[d][b] ;
    }
    theMean /= nDraws() ;
    mean.push_back(theMean) ;
  }

  varCovar = new patHybridMatrix(nBetas()) ;
  varCovar->init(0.0) ;
  for (patULong b1 = 0 ; b1 < nBetas() ; ++b1) {
    for (patULong b2 = b1 ; b2 < nBetas() ; ++b2) {
      for (patULong d = 0 ; d < nDraws() ; ++d) {
	varCovar->addElement(b1,b2,
			     (*theBayesianDraws)[d][b1] * (*theBayesianDraws)[d][b2],err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      varCovar->multElement(b1,b2,1.0/patReal(nDraws()),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      varCovar->addElement(b1,b2,-mean[b1]*mean[b2],err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }  
}
