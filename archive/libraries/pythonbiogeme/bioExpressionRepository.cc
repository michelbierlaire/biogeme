//-*-c++-*------------------------------------------------------------
//
// File name : bioExpressionRepository.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Sep 12 16:03:17 2009
//
//--------------------------------------------------------------------
#include <sstream>
#include "bioExpressionRepository.h"
#include "patDisplay.h"
#include "bioArithConstant.h"
#include "patErrMiscError.h"

bioExpressionRepository::bioExpressionRepository(patULong threadId) : oneId(patBadId), zeroId(patBadId), theThreadId(threadId) {

  // ID zero cannot be used. 
  // A previous implementation was using pointers, and zero represents
  // the NULL pointer, and cannot point to any valid expression.

  theRepository.push_back(NULL) ;


}

patULong bioExpressionRepository::addExpression(bioExpression* exp) {
  theRepository.push_back(exp) ;
  return (theRepository.size()-1) ;
}

bioExpression* bioExpressionRepository::getExpression(patULong id) {
  if (id >= theRepository.size()) {
    return NULL ;
  }
  return theRepository[id] ;
}


bioExpression* bioExpressionRepository::getExpressionByPythonID(patString pythonID){
  if(pyIDtorepID.find(pythonID) != pyIDtorepID.end()){
    return theRepository[pyIDtorepID[pythonID]];
  }else{
    return NULL;
  }
}

void bioExpressionRepository::addIDtoMap(patString pyID, patULong repID){
  pyIDtorepID[pyID] = repID;
}

patString bioExpressionRepository::printList() {
  stringstream str ;
  for (patULong i = 1 ; i < theRepository.size() ; ++i) {
    if (theRepository[i] == NULL){
      WARNING("Expression " << i << " not defined") ;
    }
    else {
      str << "[" << i << "] " << *(theRepository[i]) << " Thread: " << theRepository[i]->getThreadId() <<endl ;
    }
  }
  return patString(str.str()) ;
}

bioExpression* bioExpressionRepository::getZero()  {
  if (zeroId == patBadId) {
    bioExpression* zero = new bioArithConstant(this,patBadId,0.0) ;
    zeroId = zero->getId() ;
  }
  return getExpression(zeroId) ;
}

bioExpression* bioExpressionRepository::getOne() {
  if (oneId == patBadId) {
    bioExpression* one = new bioArithConstant(this,patBadId,1.0) ;
    oneId = one->getId() ;
  }
  return getExpression(oneId) ;
}

patULong bioExpressionRepository::getThreadId() const {
  return theThreadId ;
}

patString bioExpressionRepository::check(patError*& err) const {
  DEBUG_MESSAGE("Verifying " << theRepository.size() << " expressions for thread " << theThreadId) ;
  for (patULong i = 0 ; i < theRepository.size() ; ++i) {
    if (theRepository[i] != NULL) {
      theRepository[i]->check(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString() ;
      }
      if (theThreadId != theRepository[i]->getThreadId()) {
	stringstream str ;
	str << "Mismatch in threadid: " << theThreadId << " and " <<  theRepository[i]->getThreadId() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patString() ;
      }
    }
  }
  return patString("OK") ;
}

bioExpressionRepository* bioExpressionRepository::getCopyForThread(patULong threadId, patError*& err) {
  bioExpressionRepository* theNewRep = new bioExpressionRepository(threadId) ;
  for(vector<bioExpression*>::iterator i =  theRepository.begin() ;
      i != theRepository.end() ;
      ++i) {
    if (*i != NULL) {
      (*i)->getShallowCopy(theNewRep,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
  }
  return theNewRep ;
}


patULong bioExpressionRepository::getNbrOfExpressions() const {
  return theRepository.size() - 1 ;
}

void bioExpressionRepository::setReport(bioReporting* r) {
  for(vector<bioExpression*>::iterator i =  theRepository.begin() ;
      i != theRepository.end() ;
      ++i) {
    if (*i != NULL) {
      (*i)->setReport(r) ;
    }
  }
}
