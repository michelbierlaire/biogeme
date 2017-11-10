//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGaussHermite.cc
// Author :    Michel Bierlaire
// Date :      Sat May 21 19:25:23 2011
//
//--------------------------------------------------------------------

#include "bioArithGaussHermite.h"
#include "bioExpression.h"
#include "bioLiteralRepository.h"
#include "patDisplay.h"
bioArithGaussHermite::bioArithGaussHermite(bioExpression* e,
					   vector<patULong> derivl,
					   patULong l,
					   patBoolean wg,
patBoolean wh) : 
  withGradient(wg),
  withHessian(wh),
  theExpression(e),
  derivLiteralIds(derivl),
  literalId(l) {
}

vector<patReal> bioArithGaussHermite::getValue(patReal x, patError*& err) {
  bioLiteralRepository::the()->setRandomVariableValue(x, 
						      literalId, 
						      getThreadId(), 
						      err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return vector<patReal>() ;
  }
  vector<patReal> result ;
  if (withGradient) {
    bioFunctionAndDerivatives* fg = theExpression->getNumericalFunctionAndGradient(derivLiteralIds,withHessian,patFALSE,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patReal>() ;
    }
    result.push_back(fg->theFunction) ;
    for (patULong i = 0 ; i < fg->theGradient.size() ; ++i) {
      result.push_back(fg->theGradient[i]) ;
    }
    // It is assumed that, if the hessian is requested, so is the gradient.
    if (withHessian) {
      for (patULong i = 0 ; i < fg->theGradient.size() ; ++i) {
	for (patULong j = i ; j < fg->theGradient.size() ; ++j) {
	  patReal v = fg->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return vector<patReal>() ;
	  }
	  result.push_back(v) ;
	}
      }
    }
  }
  else{
    patReal r = theExpression->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patReal>() ;
    }
    result.push_back(r) ;
  }
  return result ;
}

patULong bioArithGaussHermite::getSize() const {
  if (withGradient) {
      patULong n = derivLiteralIds.size() ; 
    if (withHessian) {
      return 1 + n + (n * (n+1) / 2) ;
    }
    else {
      return 1 + n ;
    }
  }
  else {
    return 1 ;
  }
}

patULong bioArithGaussHermite::getThreadId() const {
  if (theExpression == NULL) {
    return patBadId ;
  }
  return theExpression->getThreadId() ;
}

