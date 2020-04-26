//-*-c++-*------------------------------------------------------------
//
// File name : bioArithElem.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:56:09 2009
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <sstream>

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"

#include "bioExpressionRepository.h"
#include "bioParameters.h"
#include "bioArithElem.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioArithCompositeLiteral.h"

bioArithElem::bioArithElem(bioExpressionRepository* rep,
			   patULong par,
                           patULong ind,
			   map<patULong,patULong> aDict,
			   patULong defExp,
			   patError*& err) : 
  bioExpression(rep,par),
  defaultExpressionId(defExp),
  indexCalculationId(ind),
  theDictionaryIds(aDict) {
  for (map<patULong,patULong>::iterator i = theDictionaryIds.begin() ;
       i != theDictionaryIds.end() ;
       ++i) {
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
  defaultExpression = theRepository->getExpression(defaultExpressionId) ;
  relatedExpressions.push_back(defaultExpression) ;
  //defaultExpression->setParent(getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  indexCalculation = theRepository->getExpression(indexCalculationId) ;
  //indexCalculation->setParent(getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  relatedExpressions.push_back(indexCalculation) ;
}

bioArithElem::~bioArithElem() {}

patString bioArithElem::getOperatorName() const {
  return ("elem") ;
}


patReal bioArithElem::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    index = indexCalculation->getValue(patFALSE, patLapForceCompute, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    map<patULong,bioExpression*>::const_iterator found = 
      theDictionary.find(patULong(index)) ;
    useDefault = (found == theDictionary.end()) ;
    if (useDefault) {
      patULong warning = bioParameters::the()->getValueInt("warnsForIllegalElements",err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      if (warning != 0) {
        WARNING("Index " << index << " not valid in expression " << *this) ;
      }
      patReal result = defaultExpression->getValue(prepareGradient, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      lastValue = result ;
    }else{
    
      // We use the selected exprsssion, not the default one.
      patReal result  = found->second->getValue(prepareGradient, currentLap, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return patReal() ;
      }
      lastValue = result ;
    }
    lastComputedLap = currentLap ;
  }
  return lastValue ;
}

bioExpression* bioArithElem::getDerivative(patULong aLiteralId, patError*& err) const {
  map<patULong,patULong> theDerivatives ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    bioExpression* theDeriv = i->second->getDerivative(aLiteralId,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    theDerivatives[i->first] = theDeriv->getId() ;
  }

  bioExpression* derivDefault = defaultExpression->getDerivative(aLiteralId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }

  bioExpression* result = new bioArithElem(theRepository,patBadId,indexCalculation->getId(),theDerivatives,derivDefault->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL;
  }
  return result ;
}

bioArithElem* bioArithElem::getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioExpression* leftClone(NULL) ;
  if (indexCalculation != NULL) {
    leftClone = indexCalculation->getDeepCopy(rep,err) ;
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

  bioExpression* defaultClone = defaultExpression->getDeepCopy(rep,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  bioArithElem* theNode = 
    new bioArithElem(rep,patBadId,leftClone->getId(),newDictionary,defaultClone->getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}

bioArithElem* bioArithElem::getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const {
  bioArithElem* theNode = 
    new bioArithElem(rep,patBadId,indexCalculationId,theDictionaryIds,defaultExpressionId,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theNode ;
}





patString bioArithElem::getExpression(patError*& err) const {
  stringstream str ;
  str << "Elem(" ;
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
    str << "{" << i->first << ": " << theExpr << "}" ;
  }
  patString theIndex = indexCalculation->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  patString theDefault = defaultExpression->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  str << "[" << theIndex << "][" << theDefault << "]" ;
  return patString(str.str()) ;
  
}


patBoolean bioArithElem::dependsOf(patULong aLiteralId) const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->dependsOf(aLiteralId)) {
      return patTRUE ;
    }
  }
  if (defaultExpression->dependsOf(aLiteralId)) {
    return patTRUE ;
  }
  return patFALSE ;
}



void bioArithElem::simplifyZeros(patError*& err) {
  if (indexCalculation != NULL) {
    if (indexCalculation->isStructurallyZero()) {
      //      DELETE_PTR(indexCalculation) ;
      indexCalculation = new bioArithConstant(theRepository,getId(),0) ;
      indexCalculationId = indexCalculation->getId() ;
    }
    else {
      indexCalculation->simplifyZeros(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  if (defaultExpression != NULL) {
    if (defaultExpression->isStructurallyZero()) {
      //      DELETE_PTR(indexCalculation) ;
      defaultExpression = theRepository->getZero() ;
      defaultExpressionId = defaultExpression->getId() ;
    }
    else {
      defaultExpression->simplifyZeros(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->isStructurallyZero()) {
      theDictionary[i->first] = theRepository->getZero() ;
    }
    else {
      i->second->simplifyZeros(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  
}

patULong bioArithElem::getNumberOfOperations() const {
  patULong result = 0 ;
  if (indexCalculation != NULL) {
    result += indexCalculation->getNumberOfOperations() ;
  }
  if (defaultExpression != NULL) {
    result += defaultExpression->getNumberOfOperations() ;
  }
  // Although it is not really correct, we consider the sum of all operations
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
      result += i->second->getNumberOfOperations() ;
  }
  return (result) ;
}

patULong bioArithElem::lastIndex() const {
  return theDictionary.rbegin()->first ;
}

patString bioArithElem::getExpressionString() const {
  stringstream str ;
  str << "$D" ;
  if (indexCalculation != NULL) {
    str << '[' << indexCalculation->getExpressionString() << ']' ;
  }
  if (defaultExpression != NULL) {
    str << '[' << defaultExpression->getExpressionString() << ']' ;
  }
  str << '[' ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    //    DEBUG_MESSAGE("BEFORE: " << *i->second) ;
    str << "{" << i->second->getExpressionString() << "}" ;
  }  
  str << ']' ;
  return patString(str.str()) ;
}

patBoolean bioArithElem::containsAnIterator() const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->containsAnIterator()) {
      return patTRUE ;
    }
  }

  if (defaultExpression->containsAnIterator()) {
    return patTRUE ;
  }
  return patFALSE ;
}

patBoolean bioArithElem::containsAnIteratorOnRows() const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->containsAnIteratorOnRows()) {
      return patTRUE ;
    }
  }

  if (defaultExpression->containsAnIteratorOnRows()) {
    return patTRUE ;
  }
  return patFALSE ;
}

patBoolean bioArithElem::containsAnIntegral() const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  if (defaultExpression->containsAnIntegral()) {
    return patTRUE ;
  }
  return patFALSE ;
}


patBoolean bioArithElem::containsASequence() const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (i->second->containsASequence()) {
      return patTRUE ;
    }
  }
  if (defaultExpression->containsASequence()) {
    return patTRUE ;
  }
  return patFALSE ;
}


void bioArithElem::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    i->second->collectExpressionIds(s) ;
  }  
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->collectExpressionIds(s) ;
  }
  
}





  

bioFunctionAndDerivatives* bioArithElem::getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,patBoolean debugDeriv, patError*& err) {
  index = indexCalculation->getValue(patFALSE, patLapForceCompute, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }

  map<patULong,bioExpression*>::const_iterator found = 
    theDictionary.find(patULong(index)) ;

  if (found == theDictionary.end()) {
    patULong warning = bioParameters::the()->getValueInt("warnsForIllegalElements",err) ;
    if (warning != 0) {
      WARNING("Index " << index << " not valid in expression " << *this) ;
    }
    bioFunctionAndDerivatives* r =
      defaultExpression->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    return r ;
  }
  else {
    bioFunctionAndDerivatives* r =
      found->second->getNumericalFunctionAndGradient(literalIds,computeHessian,debugDeriv,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    return r ;
  }
}

patBoolean bioArithElem::isStructurallyZero() const {
  for (map<patULong,bioExpression*>::const_iterator i = theDictionary.begin() ;
       i != theDictionary.end() ;
       ++i) {
    if (!i->second->isStructurallyZero()) {
      return patFALSE ;
    }
  }  
  return patTRUE ;
}
patString bioArithElem::check(patError*& err) const  {
  stringstream str ;
  if (defaultExpression->getId() != defaultExpressionId) {
    str << "Incompatible IDS for children: " << defaultExpression->getId() << " and " << defaultExpressionId << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
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
  return patString(str.str()) ;
}

