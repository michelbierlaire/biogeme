//-*-c++-*------------------------------------------------------------
//
// File name : patMemoryObservationIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Mar 29 22:55:44 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patMemoryObservationIterator.h"

patMemoryObservationIterator::patMemoryObservationIterator(vector<patIndividualData>* data) : theData(data) {
  
}
  
void patMemoryObservationIterator::first() {
  individualIter = theData->begin() ;
  if (individualIter != theData->end()) {
    obsIter = individualIter->theObservations.begin() ;
  }
}

void patMemoryObservationIterator::next() {
  if (individualIter == theData->end()) {
    return ;
  }
  ++obsIter ;
  while (obsIter == individualIter->theObservations.end()) {
    ++individualIter ;
    if (!isDone()) {
      obsIter = individualIter->theObservations.begin() ;
    }
  }
}

patBoolean patMemoryObservationIterator::isDone() {
  return (individualIter == theData->end()) ;
}

patObservationData* patMemoryObservationIterator::currentItem() {
  return &(*obsIter) ;

}

