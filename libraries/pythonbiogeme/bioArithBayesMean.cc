//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayesMean.cc
// Author :    Michel Bierlaire
// Date :      Fri Oct 19 11:14:10 2012
//
//--------------------------------------------------------------------

#include "bioArithBayesMean.h"
#include "bioExpressionRepository.h"
#include "bioParameters.h"
#include "patErrNullPointer.h"
#include "bioSample.h"
#include "patMultivariateNormal.h"
#include "bioRandomDraws.h"
#include "patNormalWichura.h"
#include "bioLiteralRepository.h"

bioArithBayesMean::bioArithBayesMean(bioExpressionRepository* rep,
				     patULong par,
				     vector<patULong> means,
				     vector<patULong> realizations,
				     vector<vector<patULong> > varcovar,
				     patError*& err) :
  bioArithBayes(rep,par,means,err),
  theRealizationsExpression(realizations),
  theVarcovarExpression(varcovar),
  theRandomDraws(NULL),
  mu(NULL),
  sigma(NULL) {
  
  individualId = bioParameters::the()->getValueString("individualId",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

bioArithBayesMean::~bioArithBayesMean() {
  if (mu != NULL) {
    DELETE_PTR(mu) ;
  }
  if (sigma != NULL) {
    DELETE_PTR(sigma) ;
  }
}

patString bioArithBayesMean::getOperatorName() const {
  return patString("BayesMean") ;
}

patString bioArithBayesMean::getExpression(patError*& err) const {

  stringstream str ;
  str << "Mean(" ;
  str << "(" ;
  for (vector<patULong>::const_iterator i = betas.begin() ;
       i != betas.end() ;
       ++i) {
    if (i != betas.begin()) {
      str << "," ;
    }
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      str << aBeta->getExpression(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString() ;
      }
    }
  }
  str << "),(" ;
  for (vector<patULong>::const_iterator i = theRealizationsExpression.begin() ;
       i != theRealizationsExpression.end() ;
       ++i) {
    if (i != theRealizationsExpression.begin()) {
      str << ","  ;
    }
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      str << aBeta->getExpression(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString() ;
      }
    }
  }
  str << "),(" ;
  for (vector<vector<patULong> >::const_iterator i = theVarcovarExpression.begin() ;
       i != theVarcovarExpression.end() ;
       ++i) {
    if (i != theVarcovarExpression.begin()) {
      str << "," ;
    }
    str << "(" ;
    for (vector<patULong>::const_iterator j = i->begin() ;
	 j != i->end() ;
	 ++j) {
      if (j!= i->begin()) {
	str << "," ;
      }
      bioExpression* aBeta = theRepository->getExpression(*j) ;
      if (aBeta != NULL) {
	str << aBeta->getExpression(err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patString() ;
	}
      }
    }
    str << ")" ;
  }
  return patString(str.str()) ;
}
  
bioArithBayesMean*  bioArithBayesMean::getDeepCopy(bioExpressionRepository* rep, 
						 patError*& err) const {
  return getShallowCopy(rep,err) ;
}

bioArithBayesMean* bioArithBayesMean::getShallowCopy(bioExpressionRepository* rep, 
						     patError*& err) const {
  bioArithBayesMean* theNode = new bioArithBayesMean(rep,
						     patBadId,
						     betas,
						     theRealizationsExpression,
						     theVarcovarExpression,
						     err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

patBoolean bioArithBayesMean::dependsOf(patULong aLiteralId) const {
  for (vector<patULong>::const_iterator i = betas.begin() ;
       i != betas.end() ;
       ++i) {
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      if (aBeta->dependsOf(aLiteralId)) {
	return patTRUE ;
      }
    }
  }

  for (vector<patULong>::const_iterator i = theRealizationsExpression.begin() ;
       i != theRealizationsExpression.end() ;
       ++i) {
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      if (aBeta->dependsOf(aLiteralId)) {
	return patTRUE ;
      }
    }
  }
  for (vector<vector<patULong> >::const_iterator i = theVarcovarExpression.begin() ;
       i != theVarcovarExpression.end() ;
       ++i) {
    for (vector<patULong>::const_iterator j = i->begin() ;
	 j != i->end() ;
	 ++j) {
      bioExpression* aBeta = theRepository->getExpression(*j) ;
      if (aBeta != NULL) {
	if (aBeta->dependsOf(aLiteralId)) {
	  return patTRUE ;
	}
      }
    }
  }
  return patFALSE ;

}

patString bioArithBayesMean::getExpressionString() const {
  stringstream str ;
  str << "BayesMean(" ;
  str << "(" ;
  for (vector<patULong>::const_iterator i = betas.begin() ;
       i != betas.end() ;
       ++i) {
    if (i != betas.begin()) {
      str << "," ;
    }
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      str << aBeta->getExpressionString() ;
    }
  }
  str << "),(" ;
  for (vector<patULong>::const_iterator i = theRealizationsExpression.begin() ;
       i != theRealizationsExpression.end() ;
       ++i) {
    if (i != theRealizationsExpression.begin()) {
      str << ","  ;
    }
    bioExpression* aBeta = theRepository->getExpression(*i) ;
    if (aBeta != NULL) {
      str << aBeta->getExpressionString() ;
    }
  }
  str << "),(" ;
  for (vector<vector<patULong> >::const_iterator i = theVarcovarExpression.begin() ;
       i != theVarcovarExpression.end() ;
       ++i) {
    if (i != theVarcovarExpression.begin()) {
      str << "," ;
    }
    str << "(" ;
    for (vector<patULong>::const_iterator j = i->begin() ;
	 j != i->end() ;
	 ++j) {
      if (j!= i->begin()) {
	str << "," ;
      }
      bioExpression* aBeta = theRepository->getExpression(*j) ;
      if (aBeta != NULL) {
	str << aBeta->getExpressionString() ;
      }
    }
    str << ")" ;
  }
  
  return patString(str.str()) ;
}

patULong bioArithBayesMean::getNumberOfOperations() const {
  return (0) ;
}



patBoolean bioArithBayesMean::containsAnIterator() const {
  return patFALSE ;
}

patBoolean bioArithBayesMean::containsAnIteratorOnRows() const {
  return patFALSE ;
}

patBoolean bioArithBayesMean::containsAnIntegral() const {
  return patFALSE ;
}

patBoolean bioArithBayesMean::containsASequence() const {
  return patFALSE ;
}

void bioArithBayesMean::simplifyZeros(patError*& err) {
}

void bioArithBayesMean::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
}



patString bioArithBayesMean::check(patError*& err) const {
  return patString() ;
}

void bioArithBayesMean::getNextDraw(patError*& err) {

  static patBoolean first = patTRUE ;

  if (first) {
    first = patFALSE ;
    prepareTheDraws(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  betaValues = theRandomDraws->getNextDraw(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  for (patULong i = 0 ; i < betaValues.size() ; ++i) {
    bioLiteralRepository::the()->setBetaValue(betasExpr[i]->getOperatorName(),
					      betaValues[i],
					      err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  return ;
}

void bioArithBayesMean::prepareTheDraws(patError*& err) {
  mu = new patVariables();
  sigma = new patHybridMatrix(betas.size()) ;
  patULong nbrIndividuals ;
  for (vector<patULong>::iterator i = theRealizationsExpression.begin() ;
       i != theRealizationsExpression.end() ;
       ++i) {
    bioExpression* beta = theRepository->getExpression(*i) ;
    if (beta == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return ;
    }
    patString theVariable = beta->getOperatorName() ;
    DEBUG_MESSAGE("Variable: " << theVariable) ;
    
    vector<patReal> values = theSample->getColumn(theVariable,individualId,err) ;
    DEBUG_MESSAGE("HERERE") ;
    nbrIndividuals = values.size() ;
    patReal sum(0.0) ;
    for (vector<patReal>::iterator i = values.begin() ;
	 i != values.end() ;
	 ++i) {
      sum += *i ;
    }
    mu->push_back(sum / patReal(nbrIndividuals)) ;
  }

  for (patULong i = 0 ; i < theVarcovarExpression.size() ; ++i) {
    for (patULong j = i ; j < theVarcovarExpression.size() ; ++j) {
      bioExpression* aBeta = 
	theRepository->getExpression(theVarcovarExpression[i][j]) ;
      if (aBeta == NULL) {
	err = new patErrNullPointer("bioExpression") ;
	WARNING(err->describe()) ;
	return ;
      }
      patReal v = aBeta->getValue(patFALSE, patLapForceCompute, err) ;
      bioExpression* symBeta = 
	theRepository->getExpression(theVarcovarExpression[j][i]) ;
      if (symBeta == NULL) {
	err = new patErrNullPointer("bioExpression") ;
	WARNING(err->describe()) ;
	return ;
      }
      patReal symv = symBeta->getValue(patFALSE, patLapForceCompute, err) ;
      if (v != symv) {
	stringstream str ;
	str << "The varcovar matrix is not symmetric. Check entries in row " << i+1 << " and column " << j+1 ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      sigma->setElement(i,j,v,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  
  patNormalWichura* theNormal = new patNormalWichura() ;
  theNormal->setUniform(bioRandomDraws::the()->getUniformGenerator()) ;
  
  DEBUG_MESSAGE("mu = " << *mu) ;
  DEBUG_MESSAGE("sigma = " << *sigma) ;
  theRandomDraws = new patMultivariateNormal(mu,
					     sigma,
					     theNormal,
					     err) ;
  
}
