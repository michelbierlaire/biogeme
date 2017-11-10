//-*-c++-*------------------------------------------------------------
//
// File name : patMultivariateNormal.cc
// Author :    Michel Bierlaire
// Date :      Sat Sep 10 17:23:00 2011
//
//--------------------------------------------------------------------

#include "patErrNullPointer.h"
#include "patMultivariateNormal.h"
#include "patHybridMatrix.h"
#include "patNormal.h"

patMultivariateNormal::patMultivariateNormal(patVariables* mu, 
					     patHybridMatrix* sigma, 
					     patRandomNumberGenerator* normalGenerator,
					     patError*& err) :
  theNormalGenerator(normalGenerator),
  theMean(mu),
  theVarCovar(sigma) {
  
  if (mu == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe());
    return ;
  }
  if (sigma == NULL) {
    err = new patErrNullPointer("patHybridMatrix") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (normalGenerator == NULL) {
    err = new patErrNullPointer("patNormal") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (mu->size() != sigma->getSize()) {
    stringstream str ;
    str << "Incompatible sizes: " << mu->size() << " and " << sigma->getSize() ;
    err = new patErrMiscError(str.str());
    WARNING(err->describe()) ;
    return ;
  }


  patBoolean defPositive = theVarCovar->straightCholesky(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (!defPositive) {
    err = new patErrMiscError("Variance-covariance matrix is not positive definite") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (!normalGenerator->isNormal()) {
    err = new patErrMiscError("Requires a generator of normal draws") ;
    WARNING(err->describe()) ;
    return ;
  }
}

patVariables patMultivariateNormal::getNextDraw(patError*& err) {

  patVariables result = *theMean ;
  patVariables independentDraws(result.size()) ;
  for (patULong i = 0 ; i < independentDraws.size() ; ++i) {
    pair<patReal,patReal> normalDraw =theNormalGenerator->getNextValue(err) ;
    independentDraws[i] = normalDraw.first ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patVariables() ;
    }
  }
  for (patULong i = 0 ; i < independentDraws.size() ; ++i) {
    for (patULong j = 0 ; j <= i ; ++j) {
      result[i] += theVarCovar->getElement(i,j,err) * independentDraws[j] ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patVariables() ;
      }
    }    
  } 
  return result ;
}

