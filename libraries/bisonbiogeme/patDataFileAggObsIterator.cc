//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileAggObsIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jan 15 11:57:42 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patDataFile.h"
#include "patDataFileAggObsIterator.h"
#include "patDataFileIndIterator.h"
#include "patString.h"

patDataFileAggObsIterator::patDataFileAggObsIterator(patDataFile* aDataFile) 
  : theDataFile(aDataFile), theIndIterator(NULL), currentIndividual(NULL) {
  
}

patDataFileAggObsIterator::~patDataFileAggObsIterator() {
  if (theIndIterator != NULL) {
    DELETE_PTR(theIndIterator) ;
  } 
}


void patDataFileAggObsIterator::first() {
  theIndIterator = theDataFile->createIndIterator() ;
  if (theIndIterator == NULL) {
    return ;
  }
  theIndIterator->first() ;
  if (!theIndIterator->isDone()) {
    currentIndividual = theIndIterator->currentItem() ;
    if (currentIndividual == NULL) {
      return ;
    }
    theAggObsIterator = currentIndividual->theAggregateObservations.begin() ;
  }
}

void patDataFileAggObsIterator::next() {
  if (currentIndividual == NULL) {
    return ;
  }
  ++theAggObsIterator ;
  while (theAggObsIterator == currentIndividual->theAggregateObservations.end()) {
    theIndIterator->next() ;
    if (!theIndIterator->isDone()) {
      currentIndividual = theIndIterator->currentItem() ;
      if (currentIndividual == NULL) {
	return ;
      }
      theAggObsIterator = currentIndividual->theAggregateObservations.begin() ;
    }
  }
}

patBoolean patDataFileAggObsIterator::isDone() {
  return theIndIterator->isDone() ;
}

patAggregateObservationData* patDataFileAggObsIterator::currentItem() {
  if (currentIndividual == NULL) {
    return NULL ;
  }
  return &(*theAggObsIterator) ;
}
