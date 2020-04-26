//-*-c++-*------------------------------------------------------------
//
// File name : bioSampleIterator.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Jul 13 15:27:42  2009
//
//--------------------------------------------------------------------

#include <iterator>
#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "bioIteratorInfoRepository.h"
#include "bioRowIterator.h"
#include "patMath.h"

bioRowIterator::bioRowIterator(const vector<patVariables>* db, 
			       bioIteratorSpan aSpan,
			       bioIteratorSpan threadSpan,
			       patError*& err) :

  theIteratorName(aSpan.name),
  dataBase(db),
  firstRow(aSpan.firstRow),
  lastRow(aSpan.lastRow),
  printLoopTime(patFALSE) ,
  theLoopTime(0) {


  rowPointers = bioIteratorInfoRepository::the()->getRowPointers(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patBoolean isTop = bioIteratorInfoRepository::the()->isTopIterator(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (isTop) {
    bioIteratorSpan intersect = aSpan.intersection(threadSpan) ;
    firstRow = intersect.firstRow ;
    lastRow = intersect.lastRow ;
  }

}

bioRowIterator::bioRowIterator(const vector<patVariables>* db, 
			       bioIteratorSpan aSpan,
			       bioIteratorSpan threadSpan,
			       patBoolean messages,
			       patError*& err) :
  theIteratorName(aSpan.name),
  dataBase(db),
  firstRow(aSpan.firstRow),
  lastRow(aSpan.lastRow),
  printLoopTime(messages) ,
  theLoopTime(aSpan.lastRow-aSpan.firstRow+1) {

  displayInterval = patULong(floor(patReal(aSpan.lastRow-aSpan.firstRow+1) / 100.0) * 100) / 10 ;
  rowPointers = bioIteratorInfoRepository::the()->getRowPointers(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patBoolean isTop = bioIteratorInfoRepository::the()->isTopIterator(theIteratorName,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (isTop) {
    bioIteratorSpan intersect = aSpan.intersection(threadSpan) ;
    firstRow = intersect.firstRow ;
    lastRow = intersect.lastRow ;
  }


}

void bioRowIterator::first() {
  currentRow = firstRow ;
}


void bioRowIterator::next() {
  if (isDone()) {
    return ;
  }
  if (currentRow != patBadId) {
    if (currentRow > rowPointers.size()) {
      WARNING("ROW NUMBER OUT OF BOUNDS: " << currentRow <<  " > " << rowPointers.size()) ;
    }
    else {
      ++currentRow ;
      //      currentRow = rowPointers[currentRow] ;
    }
  }
  if (displayInterval > 0) {
    if (printLoopTime && (currentRow % displayInterval == 0)) {
      theLoopTime.setIteration(currentRow-firstRow) ;
      GENERAL_MESSAGE(theLoopTime) ;
    }
  }
}


patBoolean bioRowIterator::isDone() {
  if (currentRow == patBadId) {
    return patTRUE ;
  }
  return (currentRow >= lastRow) ;
  
}


const patVariables* bioRowIterator::currentItem() {
  if (!this->isDone()) {
    if (dataBase == NULL) {
      WARNING(" NO DATABASE") ;
      return NULL ;
    }
    if (currentRow >= dataBase->size()) {
      return NULL ;
    }
    return &((*dataBase)[currentRow]) ;
  } 
  else {
    return NULL ;
  }
}

patULong bioRowIterator::getCurrentRow() const {
  return currentRow ;
}


patULong bioRowIterator::nbrOfItems() const {
  return dataBase->size() ;
}
