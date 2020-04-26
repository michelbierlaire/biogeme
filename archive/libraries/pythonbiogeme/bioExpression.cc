//-*-c++-*------------------------------------------------------------
//
// File name : bioExpression.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed Apr 29 14:31:38 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patMath.h"
#include "bioExpression.h"
#include "patDisplay.h"
#include "bioSample.h"
#include "patLap.h"

#include "bioParameters.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"
#include "bioLiteralRepository.h"

bioExpression::bioExpression(bioExpressionRepository* rep, patULong par) :
  parent(par),
  lastComputedLap(0),
  __x(NULL), 
  __draws(NULL),
  __unifdraws(NULL),
  theSample(NULL),
  theRepository(rep),
  theReport(NULL) {
  

  theId = theRepository->addExpression(this) ;
  performCheck = bioParameters::the()->getValueInt("moreRobustToNumericalIssues") != 0 ;
}

ostream& operator<<(ostream& str, const bioExpression& x) {
  patError* err = NULL ;
  str << x.getExpression(err) ;
  if (err != NULL) {
    str << err->describe() << '\t' ;
  }
  return str ;
}

// Memory management for the expression nodes is not well designed. So
// memory leak will appear but it is expected that it will not be
// major.
bioExpression::~bioExpression() {
  // Free memory !!!
}

patBoolean bioExpression::isTop() const {
  return parent == patBadId;
  //return top;
}

void bioExpression::setTop(patBoolean t){
  top = t;
}

void bioExpression::setSample(bioSample* s) {
  theSample = s ;
  for (vector<bioExpression*>::iterator iter = relatedExpressions.begin() ;
       iter != relatedExpressions.end() ;
       ++iter) {
    (*iter)->setSample(s) ;
  }
}

void bioExpression::setReport(bioReporting* s) {
  theReport = s ;
  for (vector<bioExpression*>::iterator iter = relatedExpressions.begin() ;
       iter != relatedExpressions.end() ;
       ++iter) {
    (*iter)->setReport(s) ;
  }
}

patString bioExpression::getInfo() const {
  stringstream str ;
  str << "[" << getId() << "]" ;
  if (isTop()) {
    str << "[" << *this << "]"  ;
    return patString(str.str());
  }
  str << "[" << *this << "]" ;
  return patString(str.str()) ;
}

patString bioExpression::getUniqueName() const {
  stringstream str ;
  str << "bio_" << theId ;
  return patString(str.str()) ;
}


patULong bioExpression::getId() const {
  return theId ;
}



patBoolean bioExpression::isStructurallyZero() const {
  return patFALSE ;
}

patBoolean bioExpression::isStructurallyOne() const {
  return patFALSE ;
}

patString bioExpression::theIterator() const {
  return patString() ;
}


patBoolean bioExpression::operator!=(const bioExpression& x) {
  return (!(*this == x)) ;
}

patBoolean bioExpression::operator==(const bioExpression& x) {
  patString me = getExpressionString() ;
  patString him = x.getExpressionString() ;
  return (me == him) ;
}



patBoolean bioExpression::isLiteral() const {
  return patFALSE ;
}


patBoolean bioExpression::isConstant() const {
  return patFALSE ;
}

patBoolean bioExpression::isSequence() const {
  return patFALSE ;
}

bioArithLikelihoodFctAndGrad* bioExpression::getLikelihoodFunctionAndDerivatives(vector<patULong> literalIds, 
										 patError*& err) const {
  err = new patErrMiscError("This function is valid only for a sum iterator.") ;
  WARNING(err->describe()) ;
  return NULL ;

}
patBoolean bioExpression::isSumIterator() const {
  return patFALSE ;
}


patBoolean bioExpression::isNamedExpression() const {
  return patFALSE ;
}

trHessian* bioExpression::getBhhh() {
  return NULL ;
}



patBoolean bioExpression::isBayesian() const {
  return patFALSE ;
}


patBoolean bioExpression::isSimulator() const {
  return patFALSE ;
}

patULong bioExpression::getThreadId() const {
  if (theRepository == NULL) {
    return patBadId ;
  }
  return theRepository->getThreadId() ;
}

bioFunctionAndDerivatives* bioExpression::getNumericalFunctionAndFinDiffGradient(vector<patULong> literalIds, patError*& err)  {
  if (findiff.empty()) {
    findiff.resize(literalIds.size()) ;
  }
  findiff.theFunction = getValue(patFALSE, patLapForceCompute, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  patReal sqrteta(patSQRT_EPSILON) ;

  patVariables beta = bioLiteralRepository::the()->getBetaValues(patFALSE) ;
  patVariables betaPlus = beta ;

  for (patULong j = 0; j < beta.size() ; ++j) {
    patReal stepsizej = sqrteta*
      patMax(patAbs(beta[j]), patOne)*patSgn(beta[j]);
    patReal tempj =  betaPlus[j];
    betaPlus[j] += stepsizej ;
    stepsizej = betaPlus[j] - tempj;
    bioLiteralRepository::the()->setBetaValues(betaPlus,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    patReal fj = getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    findiff.theGradient[j] = ((fj - findiff.theFunction)/stepsizej); 
    betaPlus[j] = tempj ;
  }
  bioLiteralRepository::the()->setBetaValues(beta,err) ; 
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return &findiff ;
  
}

vector<patULong> bioExpression::getListOfDraws(patError*& err) const {
  vector<patULong> result ;
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    vector<patULong> l = (*i)->getListOfDraws(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patULong>() ;
    }
    for (vector<patULong>::iterator iter = l.begin() ;
	 iter != l.end() ;
	 ++iter) {
      result.push_back(*iter) ;
    }
  }
  return result ;
}

void bioExpression::setDraws(pair<patReal**, patReal**> d) {
  __draws = d.first ;
  __unifdraws = d.second ;
  for (vector<bioExpression*>::iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->setDraws(d) ;
  }
}



void bioExpression::setCurrentSpan(bioIteratorSpan aSpan) {
  theCurrentSpan = aSpan ;
  for (vector<bioExpression*>::iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->setCurrentSpan(aSpan) ;
  }

}
  
void bioExpression::setThreadSpan(bioIteratorSpan aSpan) {
  theThreadSpan = aSpan ;
  for (vector<bioExpression*>::iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->setThreadSpan(aSpan) ;
  }
}

void bioExpression::setVariables(const patVariables* x) {
  __x = x ;
  for (vector<bioExpression*>::iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->setVariables(x) ;
  }
}


bioExpression* bioExpression::getParent() {
  if (theRepository == NULL) {
    return NULL ;
  }
  if (parent == patBadId) {
    return NULL ;
  }
  bioExpression* res = theRepository->getExpression(parent) ;
  return res ;
}

patBoolean bioExpression::containsMonteCarlo() const {
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    patBoolean result = (*i)->containsMonteCarlo() ;
    if (result) {
      return patTRUE ;
    }
  }
  return patFALSE ;
  
}

void bioExpression::checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) {
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->checkMonteCarlo(insideMonteCarlo,err) ;
    if (err != NULL) {
      WARNING("Expression: " << *this) ;
      WARNING(err->describe()) ;
      return ;
    }
  }
}



