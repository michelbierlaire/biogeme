//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayesMH.cc
// Author :    Michel Bierlaire
// Date :      Tue Jul 31 16:20:02 2012
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bioArithBayesMH.h"
#include "patErrNullPointer.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h"
#include "bioExpressionRepository.h"
#include "patNormalWichura.h"
#include "patUniform.h"

bioArithBayesMH::bioArithBayesMH(bioExpressionRepository* rep,
		patULong par,
		vector<patULong> theBetas,
		patULong theDensity,
		patULong theWarmup,
		patULong theSteps,
		patError*& err) :
  bioArithBayes(rep,par,theBetas,err),warmup(theWarmup), steps(theSteps) {

  
  densityExpression = theRepository->getExpression(theDensity) ;
  relatedExpressions.push_back(densityExpression) ;
  rho = bioParameters::the()->getValueReal("MetropolisHastingsNeighborhoodSize",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

bioArithBayesMH::~bioArithBayesMH() {

}

patString bioArithBayesMH::getOperatorName() const {

  return patString("BayesMH") ;
}

patString bioArithBayesMH::getExpression(patError*& err) const {
  stringstream str ;
  str << "MH(" ;
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
  str << ")," ;
  str << densityExpression->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  str << "," << warmup << "," << steps ;
  str << ")" ;
  return patString(str.str()) ;
}

bioArithBayesMH* bioArithBayesMH::getDeepCopy(bioExpressionRepository* rep, 
					      patError*& err) const {

  if (densityExpression == NULL){
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioExpression* theClone = densityExpression->getDeepCopy(rep,err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL ;
  }
  bioArithBayesMH* theNode = new bioArithBayesMH(rep,patBadId,betas,theClone->getId(),warmup,steps,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;

}

bioArithBayesMH* bioArithBayesMH::getShallowCopy(bioExpressionRepository* rep, 
						 patError*& err) const {
  if (densityExpression == NULL){
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioArithBayesMH* theNode = new bioArithBayesMH(rep,patBadId,betas,densityExpression->getId(),warmup,steps,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

patBoolean bioArithBayesMH::dependsOf(patULong aLiteralId) const  {
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
  if (densityExpression->dependsOf(aLiteralId)) {
    return patTRUE ;
  }
  return patFALSE ;


}

patString bioArithBayesMH::getExpressionString() const {
  stringstream str ;
  str << "MH(" ;
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
  str << ")," ;
  str << densityExpression->getExpressionString() ;
  str << "," << warmup << "," << steps ;
  str << ")" ;
  return patString(str.str()) ;
}




patBoolean bioArithBayesMH::updateBetas(patError*& err) {
  vector<patReal> candidate(betaValues) ;
  for (patULong i = 0 ; i < candidate.size() ; ++i) {
    pair<patReal,patReal> theDraw =theNormal->getNextValue(err) ;
    candidate[i] += rho * theDraw.first ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    bioLiteralRepository::the()->setBetaValue(betasExpr[i]->getOperatorName(),candidate[i],err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
  }
  patReal logCandidateValue = densityExpression->getValue(patFALSE, patLapForceCompute, err) ;

  if (logCandidateValue >= logCurrentValue) {
    // Accept
    betaValues = candidate ;
    logCurrentValue = logCandidateValue ;
    return patTRUE ;
  }
  else {
    patReal logMu = log(theUniform->getUniform(err)) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    if (logMu <= logCandidateValue - logCurrentValue) {
      // Accept
      betaValues = candidate ;
      logCurrentValue = logCandidateValue ;
      return patTRUE ;
    }
    else {
      return patFALSE ;
    }
  }
}

void bioArithBayesMH::getNextDraw(patError*& err) {

  static patBoolean first = patTRUE ;

  // Warmup

  if (first) {
    first = patFALSE ;
    total = 0 ;
    accept = 0 ;
    for (patULong i = 0 ; i < warmup ; ++i) {
      patBoolean acc = updateBetas(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      ++total ;
      if (acc) {
	++accept ;
      }
    }
    GENERAL_MESSAGE("Warmup done with " << 100.0 * patReal(accept) / patReal(total) << "% accepted draws") ;
  }

  for (patULong i = 0 ; i < steps ; ++i) {
    patBoolean acc = updateBetas(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    ++total ;
    if (acc) {
      ++accept ;
    }
  }
    //    DEBUG_MESSAGE("% of accept= " << 100.0 * patReal(accept) / patReal(total)) ;

}

patULong bioArithBayesMH::getNumberOfOperations() const {
  // per draw... ignoring the warmup
  if (densityExpression == NULL) {
    return 0 ;
  }
  patULong result = densityExpression->getNumberOfOperations() ;
  return (result * steps) ;
}



patBoolean bioArithBayesMH::containsAnIterator() const {
  return densityExpression->containsAnIterator() ;
}

patBoolean bioArithBayesMH::containsAnIteratorOnRows() const {
  return densityExpression->containsAnIteratorOnRows() ;
}

patBoolean bioArithBayesMH::containsAnIntegral() const {
  return densityExpression->containsAnIntegral() ;
}

patBoolean bioArithBayesMH::containsASequence() const {
  return densityExpression->containsASequence() ;
}

void bioArithBayesMH::simplifyZeros(patError*& err) {
}





void bioArithBayesMH::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  if (densityExpression != NULL) {
    densityExpression->collectExpressionIds(s) ;
  }
}



patString bioArithBayesMH::check(patError*& err) const {
  return patString() ;
}

