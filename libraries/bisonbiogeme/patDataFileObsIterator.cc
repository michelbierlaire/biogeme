//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileObsIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 15:01:44 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patDataFile.h"
#include "patDataFileObsIterator.h"
#include "patDataFileIndIterator.h"
#include "patString.h"

patDataFileObsIterator::patDataFileObsIterator(patDataFile* aDataFile) 
  : theDataFile(aDataFile), theIndIterator(NULL), currentIndividual(NULL) {
  
}

patDataFileObsIterator::~patDataFileObsIterator() {
  if (theIndIterator != NULL) {
    DELETE_PTR(theIndIterator) ;
  } 
}


void patDataFileObsIterator::first() {
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
    theObsIterator = currentIndividual->theObservations.begin() ;
  }
}

void patDataFileObsIterator::next() {
  if (currentIndividual == NULL) {
    return ;
  }
  ++theObsIterator ;
  while (theObsIterator == currentIndividual->theObservations.end()) {
    theIndIterator->next() ;
    if (!theIndIterator->isDone()) {
      currentIndividual = theIndIterator->currentItem() ;
      if (currentIndividual == NULL) {
	return ;
      }
      theObsIterator = currentIndividual->theObservations.begin() ;
    }
  }
}

patBoolean patDataFileObsIterator::isDone() {
  return theIndIterator->isDone() ;
}

patObservationData* patDataFileObsIterator::currentItem() {
  if (currentIndividual == NULL) {
    return NULL ;
  }
  return &(*theObsIterator) ;
}
