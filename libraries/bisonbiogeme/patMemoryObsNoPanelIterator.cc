//-*-c++-*------------------------------------------------------------
//
// File name : patMemoryObsNoPanelIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Mar 29 22:55:44 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patMemoryObsNoPanelIterator.h"

patMemoryObsNoPanelIterator::patMemoryObsNoPanelIterator(vector<patIndividualData>* data) : theData(data) {
  
}
  
void patMemoryObsNoPanelIterator::first() {
  individualIter = theData->begin() ;
}

void patMemoryObsNoPanelIterator::next() {
  ++individualIter ;
}

patBoolean patMemoryObsNoPanelIterator::isDone() {
  return (individualIter == theData->end()) ;
}

patObservationData* patMemoryObsNoPanelIterator::currentItem() {
  return &(*individualIter->theObservations.begin()) ;

}

