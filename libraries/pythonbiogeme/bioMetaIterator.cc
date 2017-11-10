//-*-c++-*------------------------------------------------------------
//
// File name : bioMetaIterator.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed Jun 17 16:08:37  2009
//
//--------------------------------------------------------------------


#include "patDisplay.h"
#include "bioMetaIterator.h"
#include "bioIteratorInfoRepository.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patMath.h"

bioMetaIterator::bioMetaIterator(const vector<patVariables>* db, 
				 bioIteratorSpan aSpan, 
				 bioIteratorSpan threadSpan,
				 patError*& err) :
  dataBase(db),
  theIteratorName(aSpan.name),
  firstRow(aSpan.firstRow),
  lastRow(aSpan.lastRow) {

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


void bioMetaIterator::first() {
  currentRow = firstRow ;
}


void bioMetaIterator::next() {
  if (currentRow != patBadId) {
    if (currentRow > rowPointers.size()) {
      WARNING("ROW NUMBER OUT OF BOUNDS: " << currentRow <<  " > " << rowPointers.size()) ;
    }
    else {
      currentRow = rowPointers[currentRow] ;
    }
  }
}


patBoolean bioMetaIterator::isDone() {
  if (currentRow == patBadId) {
    return patTRUE ;
  }
  return (currentRow >= lastRow) ;
}


bioIteratorSpan bioMetaIterator::currentItem() {
  bioIteratorSpan result ;
  result.name = theIteratorName ;
  
  if (!this->isDone()) {
    result.firstRow = currentRow ; 
    if (currentRow == patBadId) {
      result.lastRow = patBadId ;
    }
    else {
      result.lastRow = rowPointers[currentRow] ;
    }
  } 
  else {
    result.firstRow = patBadId  ;
    result.lastRow = patBadId  ;
  }
  return result ;
}


const patVariables* bioMetaIterator::getFirstRow() {
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

patULong bioMetaIterator::nbrOfItems() const {
  return rowPointers.size() ;
}
