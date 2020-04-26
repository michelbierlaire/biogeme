//-*-c++-*------------------------------------------------------------
//
// File name : bioDrawIterator.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 31 13:24:39 2009
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioDrawIterator.h"

bioDrawIterator::bioDrawIterator(patReal*** db, patReal*** un, patULong R) :
  dataBase(db), uniform(un), nDraws(R) {

}
void bioDrawIterator::first() {
  currentRow = 0 ;
}
void bioDrawIterator::next() {
  ++currentRow ;
}

patBoolean bioDrawIterator::isDone() {
  return (currentRow >= nDraws) ;
}

pair<patReal**,patReal**> bioDrawIterator::currentItem() {
  if (!this->isDone()) {
    if (dataBase == NULL) {
      WARNING(" NO DATABASE") ;
      return pair<patReal**,patReal**>(NULL,NULL) ;
    }
    //    if (currentRow >= dataBase->size()) {
    if (currentRow >= nDraws) {
      return pair<patReal**,patReal**>(NULL,NULL) ;
    }
    //    return &((*dataBase)[currentRow]) ;
    if (uniform != NULL) {
      return pair<patReal**,patReal**>(dataBase[currentRow],uniform[currentRow]) ;
    }
    else {
      return pair<patReal**,patReal**>(dataBase[currentRow],NULL) ;

    }
    
  } 
  else {
      return pair<patReal**,patReal**>(NULL,NULL) ;
  }
  
}

patULong bioDrawIterator::nbrOfItems() const {
  return nDraws ;
}

patULong bioDrawIterator::getNumberOfDraws() const {
  return nDraws ;
}
