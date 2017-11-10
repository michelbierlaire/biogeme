//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayes.cc
// Author :    Michel Bierlaire
// Date :      Wed Aug  1 10:23:59 2012
//
//--------------------------------------------------------------------

#include <iterator>

#include "patNormalWichura.h"
#include "patUniform.h"
#include "bioArithBayes.h"
#include "bioExpressionRepository.h"
#include "bioRandomDraws.h"
#include "bioParameters.h"
#include "patLoopTime.h"

bioArithBayes::bioArithBayes(bioExpressionRepository* rep,
			     patULong par,
			     vector<patULong> theBetas,
			     patError*& err)  :
  bioExpression(rep,par),betas(theBetas) {
  for (vector<patULong>::iterator i = betas.begin() ;
       i != betas.end() ;
       ++i) {
    bioExpression* bexpr = theRepository->getExpression(*i) ;
    if (bexpr == NULL) {
      stringstream str ;
      str << "No expression with ID " << *i ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    betasExpr.push_back(bexpr) ;
    betaNames.push_back(bexpr->getOperatorName()) ;
  }

  theUniform = bioRandomDraws::the()->getUniformGenerator() ;
  theNormal = new patNormalWichura() ;
  theNormal->setUniform(theUniform) ;
  
}


patReal bioArithBayes::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  err = new patErrMiscError("No value can be computed for this expression.") ;
  WARNING(err->describe()) ;
  return patReal()  ;
}

bioFunctionAndDerivatives* bioArithBayes::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  err = new patErrMiscError("No derivative can be computed for this expression.") ;
  WARNING(err->describe()) ;
  return NULL  ;
}

bioExpression* bioArithBayes::getDerivative(patULong aLiteralId, 
					      patError*& err) const {
  err = new patErrMiscError("No derivative can be computed for this expression.") ;
  WARNING(err->describe()) ;
  return NULL  ;
}


patBoolean bioArithBayes::isBayesian() const {
  return patTRUE ;
}

bioBayesianResults bioArithBayes::generateDraws(patError*& err) {

  // Initialize the beta
  betaValues.erase(betaValues.begin(),betaValues.end()) ;
  for (vector<bioExpression*>::iterator i = betasExpr.begin() ;
       i != betasExpr.end() ;
       ++i) {
    DEBUG_MESSAGE("Expression: " << *(*i)) ;
    DEBUG_MESSAGE(*(*i) << " = " << (*i)->getValue(patFALSE, patLapForceCompute, err) );
    betaValues.push_back((*i)->getValue(patFALSE, patLapForceCompute, err) );
    if (err != NULL) {
      WARNING(err->describe()) ;
      return bioBayesianResults() ;
    }
  }

  // Compute the current value of the density


  patULong nDraws = bioParameters::the()->getValueInt("NbrOfDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return bioBayesianResults() ;
  }
  patLoopTime loopTime(nDraws) ;

  while (theDraws.size() < nDraws) {
    if (theDraws.size() % 100 == 0) {
      loopTime.setIteration(theDraws.size()) ;
      DEBUG_MESSAGE(loopTime) ;
    }
    getNextDraw(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return bioBayesianResults();
    }
    theDraws.push_back(betaValues) ;
  }
  return bioBayesianResults(&theDraws,betaNames) ;
}

