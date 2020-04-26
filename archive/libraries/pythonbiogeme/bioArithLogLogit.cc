//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLogLogit.cc
// Author :    Michel Bierlaire
// Date :      Fri Jul 31 14:55:14 2015
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef DEBUG
#include "bioParameters.h"
#endif
#include "patMath.h"
#include "bioArithLogLogit.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"
#include "bioExpressionRepository.h"
#include "bioLiteralRepository.h"
#include "bioArithCompositeLiteral.h"
#include "bioArithElem.h"
#include "bioArithBinaryMinus.h"
#include "bioArithUnaryMinus.h"
#include "bioArithExp.h"
#include "bioArithMult.h"
#include "bioArithDivide.h"
#include "bioArithMultinaryPlus.h"

bioArithLogLogit::bioArithLogLogit(bioExpressionRepository* rep,
				   patULong par,
				   patULong ind,
				   map<patULong,patULong> aDict,
				   map<patULong,patULong> avDict,
				   patBoolean ll,
				   patError*& err) :
  bioExpression(rep,par),
  indexCalculationId(ind),
  theDictionaryIds(aDict),
  theAvailDictIds(avDict),
  loglogit(ll)
 {

  
  if (theDictionaryIds.size() != theAvailDictIds.size()) {
    stringstream str ;
    str << "Incompatible dictionaries in logit. Size " 
	<< theDictionaryIds.size() << " and " << theAvailDictIds.size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  
  // Maximum value of the arguent of the exponential.

  for (map<patULong,patULong>::iterator i = theDictionaryIds.begin() ;
       i != theDictionaryIds.end() ;
       ++i) {
    // Check if both dictionaries are compatible
    map<patULong,patULong>::iterator found = theAvailDictIds.find(i->first) ;
    if (found == theAvailDictIds.end()) {
      stringstream str ;
      str << "Logit: key " << i->first << " not present in availability dictionary" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    bioExpression* theExpression = theRepository->getExpression(i->second) ;
    if (theExpression != NULL) {
      theDictionary[i->first] = theExpression ;
      relatedExpressions.push_back(theExpression) ;
      //theExpression->setParent(getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else {
      stringstream str ;
      str << "Expression " << i->second << " unknown" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
  }
  for (map<patULong,patULong>::iterator i = theAvailDictIds.begin() ;
       i != theAvailDictIds.end() ;
       ++i) {
    // Check if both dictionaries are compatible
    map<patULong,patULong>::iterator found = theDictionaryIds.find(i->first) ;
    if (found == theDictionaryIds.end()) {
      stringstream str ;
      str << "Logit: key " << i->first << " not present in utility dictionary" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
    bioExpression* theExpression = theRepository->getExpression(i->second) ;
    if (theExpression != NULL) {
      theAvailDict[i->first] = theExpression ;
      relatedExpressions.push_back(theExpression) ;
      //theExpression->setParent(getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else {
      stringstream str ;
      str << "Expression " << i->second << " unknown" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return ;
    }
  }

  indexCalculation = theRepository->getExpression(indexCalculationId) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  relatedExpressions.push_back(indexCalculation) ;
}

bioArithLogLogit::~bioArithLogLogit() {

}

patString bioArithLogLogit::getOperatorName() const {
  return ("loglogit") ;
}

patReal bioArithLogLogit::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (indexCalculation == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    index = patULong(indexCalculation->getValue(patFALSE, currentLap, err));
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    vector<patBoolean> avail ;
    for (map<patULong,bioExpression*>::const_iterator i = theAvailDict.begin() ;
         i != theAvailDict.end() ;
         ++i) {
      if (i->second == NULL) {
        stringstream str ;
        str << "Null availability expression in dictionary with entry " << i->first ;
        err = new patErrMiscError(str.str()) ;
        WARNING(err->describe()) ;
        return patReal() ;
      }
      patReal a = i->second->getValue(patFALSE, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      avail.push_back(a != 0.0) ;
    }
    

    Vi.erase(Vi.begin(),Vi.end());
    patULong iChosen(patBadId) ;
    patReal chosen ;
    patReal largest(-patMaxReal) ;
    
    patULong k = 0 ;
    for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
	 i != theDictionary.end() ;
	 ++i) {
      if (avail[k]) {
	patReal V = i->second->getValue(patFALSE,currentLap,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	if (i->first == index) {
	  chosen = V ;
	  iChosen = Vi.size() ;
	}
	if (V > largest) {
	  largest = V ;
	}
	Vi.push_back(V) ;
      }
      ++k ;
    }

    maxexp = patCeil(largest / 10.0) * 10.0 ;
    for (patULong k = 0 ; k < Vi.size() ; ++k) {
	Vi[k] -= maxexp ;
    }
  
    if (iChosen == patBadId) {
      stringstream str ;
      map<patULong,bioExpression*>::iterator found = theDictionary.find(index) ;
      if (found == theDictionary.end()) {
        stringstream str ;
        str << "Alternative " << index << " unknown in dictionary." ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
      else {
	if (loglogit) {
	  stringstream str ;
	  str << "Chosen alternative " << index << " is not available" ;
	  err = new patErrMiscError(str.str()) ;
	  WARNING(*this) ;
	  WARNING(err->describe()) ;
	  return patReal() ;
	}
	else {
	  return 0.0 ;
	}
      }
    } else{
    
      // At this point, Vi contains the utilities of the available alternatives
      
      patReal denominator(0.0) ;
      for (vector<patReal>::iterator i = Vi.begin() ;
           i != Vi.end() ;
           ++i) {
        denominator += exp(*i) ;
      }
      if (loglogit) {
	lastValue = Vi[iChosen] - log(denominator) ;
      }
      else {
	if (performCheck) {
	  lastValue = patExp(Vi[iChosen]) / denominator ;
	}
	else {
	  lastValue = exp(Vi[iChosen]) / denominator ;
	}
      }
    }
    lastComputedLap = currentLap;
    return lastValue;
    
  } else{
    return lastValue;
  }
}

bioExpression* bioArithLogLogit::getDerivative(patULong aLiteralId, patError*& err) const {

  bioExpression* zero = theRepository->getZero() ;

  map<patULong,bioExpression*> deriv ;
  map<patULong,patULong> derivIds ;
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    deriv[e->first] = e->second->getDerivative(aLiteralId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    derivIds[e->first] = deriv[e->first]->getId() ;
  }
  bioArithElem* theDerivChosen = new bioArithElem(theRepository,
						  patBadId,
						  indexCalculation->getId(),
						  derivIds,
						  zero->getId(),
						  err);
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }


  // Denominator
  vector<patULong> theSum ;
  vector<patULong> theSumDeriv ;


  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {

    bioArithExp* theExp = 
      new bioArithExp(theRepository,patBadId,e->second->getId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }

    map<patULong,patULong>::const_iterator found = theAvailDictIds.find(e->first) ;
    if (found == theAvailDictIds.end()) {
      err = new patErrMiscError("Incompatible dictionaries") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    
    bioArithMult* theAvailExp = 
      new bioArithMult(theRepository,patBadId,theExp->getId(),found->second,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    theSum.push_back(theAvailExp->getId()) ;

    bioArithMult* theDerivTerm =
      new bioArithMult(theRepository,
		       patBadId,
		       theAvailExp->getId(),
		       derivIds[e->first],
		       err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    theSumDeriv.push_back(theDerivTerm->getId()) ;
  }
  bioArithMultinaryPlus* theDenom = new bioArithMultinaryPlus(theRepository,patBadId,theSum,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  bioArithMultinaryPlus* theDerivSum = 
    new bioArithMultinaryPlus(theRepository,
			      patBadId,
			      theSumDeriv,
			      err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  bioArithDivide* theRatio = new bioArithDivide(theRepository,
						patBadId,
						theDerivSum->getId(),
						theDenom->getId(),
						err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  bioArithBinaryMinus* result = new bioArithBinaryMinus(theRepository,
							patBadId,
							theDerivChosen->getId(),
							theRatio->getId(),
							err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (loglogit) {
    return result ;
  }
  bioArithLogLogit* proba = new bioArithLogLogit(theRepository,
						 patBadId,
						 indexCalculation->getId(),
						 theDictionaryIds,
						 theAvailDictIds,
						 patFALSE,
						 err);
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  bioArithMult* finalresult = new bioArithMult(theRepository,
					       patBadId,
					       proba->getId(),
					       result->getId(),
					       err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return finalresult ;
}

bioArithLogLogit* bioArithLogLogit::getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const {
  bioExpression* indexClone(NULL) ;
  if (indexCalculation != NULL) {
    indexClone = indexCalculation->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  map<patULong,patULong> newDictionary ;
  
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    bioExpression* clone = e->second->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    newDictionary[e->first] = clone->getId() ;
  }

  map<patULong,patULong> newAv ;
  
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    bioExpression* clone = a->second->getDeepCopy(rep,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    newAv[a->first] = clone->getId() ;
  }

  bioArithLogLogit* theNode = 
    new bioArithLogLogit(rep,patBadId,indexClone->getId(),newDictionary,newAv,loglogit,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;

}


bioArithLogLogit* bioArithLogLogit::getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const {
  bioArithLogLogit* theNode = 
    new bioArithLogLogit(rep,patBadId,indexCalculationId,theDictionaryIds,theAvailDictIds,loglogit,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;

}
patString bioArithLogLogit::getExpression(patError*& err) const {
  stringstream str ;
  if (loglogit) {
    str << "LogLogit" ;
  }
  else {
    str << "Logit" ;
  }
  str << "[" << *indexCalculation << "]" ;
  str << "{" ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end();
       ++i) {
    if (i != theDictionary.begin()) {
      str << "," ;
    }
    patString theExpr = i->second->getExpression(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
    map<patULong,bioExpression*>::const_iterator found =
      theAvailDict.find(i->first) ;
    patString theAv = found->second->getExpression(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
    str << i->first << ": " << theExpr << "(Av: " << theAv << ")"  ;
  }
  str << "}" ;
  return patString(str.str()) ;


}


patBoolean bioArithLogLogit::dependsOf(patULong aLiteralId) const {
  if (indexCalculation->dependsOf(aLiteralId)) {
    return patTRUE ;
  }
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    if (e->second->dependsOf(aLiteralId)) {
      return patTRUE ;
    }
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    if (a->second->dependsOf(aLiteralId)) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}

void bioArithLogLogit::simplifyZeros(patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;

}

patULong bioArithLogLogit::getNumberOfOperations() const {
  return 0 ;
}

patULong bioArithLogLogit::lastIndex() const {
  return theDictionary.rbegin()->first ;
}

patString bioArithLogLogit::getExpressionString() const {
  stringstream str ;
  str << "LogL" ;
  str << "[" << indexCalculation->getExpressionString() << "]" ;
  str << "{" ;
  for (map<patULong,bioExpression*>::const_iterator i = theAvailDict.begin() ;
       i != theAvailDict.end() ;
       ++i) {
    str << i->second->getExpressionString() ;
  }
  str << "}" ;
  str << "{" ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    str << i->second->getExpressionString() ;
  }
  str << "}" ;
  return patString(str.str()) ;
}

patBoolean bioArithLogLogit::containsAnIntegral() const {
  if (indexCalculation->containsAnIntegral()) {
    return patTRUE ;
  }
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    if (e->second->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    if (a->second->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}
patBoolean bioArithLogLogit::containsAnIterator() const {
  if (indexCalculation->containsAnIterator()) {
    return patTRUE ;
  }
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    if (e->second->containsAnIterator()) {
      return patTRUE ;
    }
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    if (a->second->containsAnIterator()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}

patBoolean bioArithLogLogit::containsAnIteratorOnRows() const {
  if (indexCalculation->containsAnIteratorOnRows()) {
    return patTRUE ;
  }
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    if (e->second->containsAnIteratorOnRows()) {
      return patTRUE ;
    }
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    if (a->second->containsAnIteratorOnRows()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}


patBoolean bioArithLogLogit::containsASequence() const {
  if (indexCalculation->containsASequence()) {
    return patTRUE ;
  }
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    if (e->second->containsASequence()) {
      return patTRUE ;
    }
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    if (a->second->containsASequence()) {
      return patTRUE ;
    }
  }
  return patFALSE ;
}


void bioArithLogLogit::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  indexCalculation->collectExpressionIds(s) ;
  for (map<patULong,bioExpression*>::const_iterator e = theDictionary.begin() ;
       e != theDictionary.end() ;
       ++e) {
    e->second->collectExpressionIds(s) ;
  }
  for (map<patULong,bioExpression*>::const_iterator a = theAvailDict.begin() ;
       a != theAvailDict.end() ;
       ++a) {
    a->second->collectExpressionIds(s) ;
  }
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->collectExpressionIds(s) ;
  }
}



bioFunctionAndDerivatives* bioArithLogLogit::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv,patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
  }
  if (logitresult.empty()) {
    logitresult.resize(literalIds.size()) ;
  }


  if (indexCalculation == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  index = patULong(indexCalculation->getValue(patFALSE, patLapForceCompute, err));
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  vector<patBoolean> avail ;
  for (map<patULong,bioExpression*>::const_iterator i = theAvailDict.begin() ;
       i != theAvailDict.end() ;
       ++i) {
    patReal a = i->second->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (a == 0.0 && i->first == index) {
      err = new patErrMiscError("%%%%%%%%%%% Chosen alternative unavailable %%%%%%%%%%") ;
      WARNING(err->describe()) ;
      return NULL;
    } 
    avail.push_back(a != 0.0) ;
  }
    
  Vig.erase(Vig.begin(),Vig.end()) ;
  patULong iChosen(patBadId) ;
  patULong iLargest(patBadId) ;
  bioFunctionAndDerivatives* chosen ;
  bioFunctionAndDerivatives* largest ;
    
  patULong k = 0 ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (avail[k]) {
      bioFunctionAndDerivatives* V = i->second->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      if (i->first == index) {
	chosen = V ;
	iChosen = Vig.size() ;
      }
      if (iLargest == patBadId) {
	largest = V ;
	iLargest = Vig.size() ;
      }
      else {
	if (V->theFunction > largest->theFunction) {
	  largest = V ;
	  iLargest = Vig.size() ;
	}
      }
      Vig.push_back(V) ;
    }
    else {
      if (i->first == index) {
	stringstream str ;
	str << "Alternative " << i->first << " is chosen but not available" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return NULL ;
      }      
    }
    ++k ;
  }

  maxexp = patCeil(largest->theFunction / 10.0) * 10.0 ;

  for (patULong k = 0 ; k < Vig.size() ; ++k) {
    Vig[k]->theFunction -= maxexp ;
  }
   
  if (iChosen == patBadId) {
    stringstream str ;
    str << "Chosen alternative not available. List of alternatives: " ;
    patULong k = 0 ;
    for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
	 i != theDictionary.end() ;
	 ++i) {
      if (avail[k]) {
	str << i->first << " " ;
      }
      else {
	str << "(" << i->first << ") " ;
	
      }
      ++k ;
    }
    str << " --- Choice: " << index ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL;
  }
  
  // At this point, Vig contains the utilities (and their
  // derivatives) of the available alternatives

  vector<patReal> expi(Vig.size()) ; ;
    
 
  patReal denominator(0.0) ;
  for (patULong k = 0 ; k < Vig.size() ; ++k) {
    expi[k] = exp(Vig[k]->theFunction) ;
    denominator += expi[k] ;
  }
  if (patAbs(denominator) < patEPSILON) {
    result.theFunction = -patMaxReal ;
    if (!loglogit) {
      logitresult.theFunction = patMaxReal ;
    }
  }
  else {
    result.theFunction = Vig[iChosen]->theFunction - log(denominator) ;
    if (!loglogit) {
      if (performCheck) {
	logitresult.theFunction = patExp(Vig[iChosen]->theFunction) / denominator ;
      }
      else {
	logitresult.theFunction = exp(Vig[iChosen]->theFunction) / denominator ;
      }
    }
  }
  vector<patReal> weightedSum(literalIds.size(),0.0) ;
  if (result.theFunction == patMaxReal) {
    for (patULong j = 0 ; j < literalIds.size() ; ++j) {
      result.theGradient[j] = patMaxReal ;
    }
  }
  else {
    for (patULong j = 0 ; j < literalIds.size() ; ++j) {
      for (patULong k = 0 ; k < Vig.size() ; ++k) {
	if (Vig[k]->theGradient[j] != 0.0) {
	  weightedSum[j] += Vig[k]->theGradient[j] * expi[k] ;
	}
      }
      result.theGradient[j] = Vig[iChosen]->theGradient[j] ;
      if (weightedSum[j] != 0.0) {
	result.theGradient[j] -= weightedSum[j] /denominator ;
      }
    }    
  }
  if (performCheck) {
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (!patFinite(result.theGradient[i])) {
	result.theGradient[i] = patMaxReal ;
      }
    }
  }

  if (!loglogit) {
    for (patULong j = 0 ; j < literalIds.size() ; ++j) {
      if (result.theGradient[j] == 0.0) {
	logitresult.theGradient[j] = 0.0 ;
      }
      else {
	logitresult.theGradient[j] = result.theGradient[j] * logitresult.theFunction ;
      }
    }
  }

  if (result.theHessian != NULL && computeHessian) {
    patReal dsquare = denominator * denominator ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      for (patULong j = i ; j < literalIds.size() ; ++j) {
	patReal dsecond(0.0) ;
	for (patULong k = 0 ; k < Vig.size() ; ++k ) {
	  if (Vig[k]->theGradient[i] != 0 && Vig[k]->theGradient[j] != 0.0) {
	    dsecond += expi[k] * Vig[k]->theGradient[i] * Vig[k]->theGradient[j] ;
	  }
	  patReal vih = Vig[k]->theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  if (vih != 0.0) {
	    dsecond += expi[k] * vih ;
	  }
	}
	patReal v =  Vig[iChosen]->theHessian->getElement(i,j,err) ;
	patReal v1(0.0) ;
	if (weightedSum[i] != 0.0 && weightedSum[j] != 0.0) {
	  v1 = weightedSum[i] * weightedSum[j] / dsquare ;
	}
	patReal v2 = dsecond / denominator ;
	result.theHessian->setElement(i,j,v+v1-v2,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
	if (!loglogit) {
	  patReal term1(0.0) ;
	  if (logitresult.theGradient[i] != 0.0 && 
	      result.theGradient[j] != 0.0) {
	    term1 = logitresult.theGradient[i] * result.theGradient[j] ;
	  }
	  patReal term2 = result.theHessian->getElement(i,j,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	  if (term2 != 0.0) {
	    term2 *= logitresult.theFunction ;
	  }
	  logitresult.theHessian->setElement(i,j,term1+term2,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}
      }
    }
  }

  // // For debug only 
   // bioFunctionAndDerivatives* findiff = getNumericalFunctionAndFinDiffGradient(literalIds, err)  ;
   // if (err != NULL) {
   //   WARNING(err->describe()) ;
   //   return NULL ;
   // }
   // patVariables error = result.theGradient - findiff->theGradient ;
   // patReal norm = norm2(error) ;
   // if (!finite(norm) || norm > 1.0e-3) {
   //   DEBUG_MESSAGE("Gradient : " << result.theGradient) ;
   //   DEBUG_MESSAGE("Fin diff.: " << findiff->theGradient) ;
   //   DEBUG_MESSAGE("Error:     " << error) ;
   // }

   //  DEBUG_MESSAGE("Logit derivatives: " << result) ;


  if (loglogit) {
#ifdef  DEBUG
    if (debugDeriv != 0) {
      bioFunctionAndDerivatives* findiff = 
	getNumericalFunctionAndFinDiffGradient(literalIds, err)  ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      patReal compare = result.compare(*findiff,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      patReal tolerance = bioParameters::the()->getValueReal("toleranceCheckDerivatives",err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      if (compare >= tolerance) {
	DEBUG_MESSAGE("*** Not shifted ***") ;
	for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
	     i != theDictionary.end() ;
	     ++i) {
	  if (avail[k]) {
	    bioFunctionAndDerivatives* V = i->second->getNumericalFunctionAndGradient(literalIds,computeHessian,patFALSE,err) ;
	    DEBUG_MESSAGE(*V) ;
	  }
	  ++k ;
	}

	DEBUG_MESSAGE("*** Shifted ***") ;
	for (patULong k = 0 ; k < Vig.size() ; ++k) {
	  DEBUG_MESSAGE("Vig[" << k << "]=" << *Vig[k]) ;
	}
	DEBUG_MESSAGE("iChosen: " << iChosen) ;
	DEBUG_MESSAGE("iLargest: " << iLargest) ;
	DEBUG_MESSAGE("Result: " << result.theGradient[3]) ;
	DEBUG_MESSAGE("Analytical: " << result) ;
	DEBUG_MESSAGE("FinDiff   : " << *findiff) ;
	WARNING("Error " << compare << " in " << *this);
	err = new patErrMiscError("Error with derivatives") ;
	WARNING(err->describe()) ;
	exit(-1);
	return NULL ;
      }
    }
#endif
    return &result ;
  }
  else {
#ifdef  DEBUG
    if (debugDeriv != 0) {
      bioFunctionAndDerivatives* findiff = 
	getNumericalFunctionAndFinDiffGradient(literalIds, err)  ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
      patReal compare = logitresult.compare(*findiff,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      patReal tolerance = bioParameters::the()->getValueReal("toleranceCheckDerivatives",err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    
      if (compare >= tolerance) {
	DEBUG_MESSAGE("Analytical: " << logitresult) ;
	DEBUG_MESSAGE("FinDiff   : " << *findiff) ;
	WARNING("Error " << compare << " in " << *this);
	err = new patErrMiscError("Error with derivatives") ;
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
#endif
    return &logitresult ;
  }
}

patString bioArithLogLogit::check(patError*& err) const  {
  stringstream str ;
  if (indexCalculation->getId() != indexCalculationId) {
      str << "Incompatible IDS for children: " << indexCalculation->getId() << " and " << indexCalculationId << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    map<patULong,patULong>::const_iterator found = theDictionaryIds.find(i->first) ;
    if (found == theDictionaryIds.end()) {
      str << "Incompatible data structure for bioArithElem: expression " << i->first << " does not have an associated ID" << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
    }
    if (i->second->getId() != found->second) {
      str << "Incompatible IDS for children: " << i->second->getId() << " and " << found->second << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
  }

  for (map<patULong,bioExpression*>::const_iterator i = theAvailDict.begin() ;
       i != theAvailDict.end() ;
       ++i) {
    map<patULong,patULong>::const_iterator found = theAvailDictIds.find(i->first) ;
    if (found == theAvailDictIds.end()) {
      str << "Incompatible data structure for bioArithElem: expression " << i->first << " does not have an associated ID" << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
    if (i->second->getId() != found->second) {
      str << "Incompatible IDS for children: " << i->second->getId() << " and " << found->second << endl ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patString() ;
    }
  }

  return patString(str.str()) ;
}


