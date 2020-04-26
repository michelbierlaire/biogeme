//-*-c++-*------------------------------------------------------------
//
// File name : bioIdIterator.cc
// Author :    Michel Bierlaire
// Date :      Tue May 19 18:22:45 2015
//
//--------------------------------------------------------------------

#include "bioIdIterator.h"
#include "patDisplay.h"

bioIdIterator::bioIdIterator(const vector<patVariables>* db, 
			     bioIteratorSpan aSpan,
			     bioIteratorSpan threadSpan,
			     patULong col,
			     patError*& err) :
  bioRowIterator(db,aSpan,threadSpan,err), column(col) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}
  
void bioIdIterator::first() {
  currentRow = firstRow ;
}

void bioIdIterator::next() {
  if (isDone()) {
    return ;
  }
  if (currentRow != patBadId) {
    if (currentRow < rowPointers.size()) {
      patULong id = (*dataBase)[currentRow][column] ;
      while (currentRow < lastRow && (*dataBase)[currentRow][column] == id) {
	++currentRow ;
      }
    }
  }
}

patBoolean bioIdIterator::isDone() {
    
  if (currentRow == patBadId) {
    return patTRUE ;
  }
  return (currentRow >= lastRow) ;
}
const patVariables* bioIdIterator::currentItem() {
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
