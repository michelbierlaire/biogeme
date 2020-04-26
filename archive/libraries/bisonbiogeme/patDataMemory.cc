//-*-c++-*------------------------------------------------------------
//
// File name : patDataMemory.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 10:50:59 2003
//
// Class storing the data in memory.
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patModelSpec.h"
#include "patDataMemory.h"
#include "patVectorIterator.h"
#include "patVectorPtrIterator.h"
#include "patSubvectorIterator.h"
#include "patSubvectorPtrIterator.h"
#include "patIndividualData.h"
#include "patObservationData.h"
#include "patMemoryObservationIterator.h"
#include "patMemoryObsNoPanelIterator.h"
#include "patShuffleVector.h"

patDataMemory::patDataMemory() : 
  indIter(NULL) ,
  obsIter(NULL) ,
  aggObsIter(NULL)
{

}

patDataMemory::~patDataMemory() {

  // These calls create a segmentation fault. I do not know why. Must
  // be investigated.

  //  DELETE_PTR(obsIter) ;
  //  DELETE_PTR(indIter) ;
}


void patDataMemory::push_back(patIndividualData* data) {
  if (data == NULL) return ;
  sample.push_back(*data) ;
}

unsigned long patDataMemory::getSize() {
  return sample.size() ;
}

void patDataMemory::erase() {
  sample.erase(sample.begin(),sample.end()) ;  
}

patIterator<patIndividualData*>* patDataMemory::createIndIterator() {
  if (indIter == NULL) {
    indIter = new patVectorIterator<patIndividualData>(&sample);
  }
  return indIter ;
}

void patDataMemory::releaseIndIterator() {
  DELETE_PTR(indIter)
}

patIterator<patObservationData*>* patDataMemory::createObsIterator() {
  // For testing
  if (obsIter == NULL) {
    obsIter = new patVectorPtrIterator<patObservationData>(&observations);
  }
  return obsIter ;
}

void patDataMemory::releaseObsIterator() {

  DELETE_PTR(obsIter) ;
}

vector<patIterator<patObservationData*>*> 
patDataMemory::createObsIteratorThread(unsigned int nbrThreads, 
				       patError*& err) {
  if (obsIterThread.size() != nbrThreads) {
    obsIterThread.resize(nbrThreads,NULL) ;
  }
  if (obsIterThread[0] == NULL) {
    for(int i = 0; i< nbrThreads; ++i) {
      patULong s = patULong(patReal(i)*patReal(observations.size())/patReal(nbrThreads)) ;
      patULong e = patULong(patReal(i+1)*patReal(observations.size())/patReal(nbrThreads)) ;
      obsIterThread[i] = 
	new patSubvectorPtrIterator<patObservationData>
	(&observations,s,e) ;
    }
  }
  return obsIterThread ;
}

void patDataMemory::releaseObsIteratorThread() {
  for (vector<patIterator<patObservationData*>*>::iterator iter = obsIterThread.begin() ;
       iter != obsIterThread.end() ;
       ++iter) {
    DELETE_PTR(*iter) ;
  }
}

vector<patIterator<patIndividualData*>*> 
patDataMemory::createIndIteratorThread(unsigned int nbrThreads, 
				       patError*& err) {
  if (indIterThread.size() != nbrThreads) {
    indIterThread.resize(nbrThreads,NULL) ;
  }

  if (indIterThread[0] == NULL) {
    for(int i=0; i< nbrThreads; ++i) {
      patULong s = patULong(patReal(i)*patReal(sample.size())/patReal(nbrThreads)) ;
      patULong e = patULong(patReal(i+1)*patReal(sample.size())/patReal(nbrThreads)) ;
      indIterThread[i] = 
	new patSubvectorIterator<patIndividualData>
	(&sample,s,e) ;
    }
  }
  return indIterThread ;
}

void patDataMemory::releaseIndIteratorThread() {
  for (vector<patIterator<patIndividualData*>*>::iterator iter = indIterThread.begin() ;
       iter != indIterThread.end() ;
       ++iter) {
    DELETE_PTR(*iter) ;
  }
}

patIterator<patAggregateObservationData*>* patDataMemory::createAggObsIterator() {
  if (aggObsIter == NULL) {
    aggObsIter = new patVectorPtrIterator<patAggregateObservationData>(&aggObservations);
  }
  return aggObsIter ;

}

void patDataMemory::releaseAggObsIterator() {
  DELETE_PTR(aggObsIter) ;
}

void patDataMemory::reserveMemory(unsigned long size, patIndividualData* instance) {
  if (instance == NULL) return ;
  sample.resize(size,*instance) ;
}

void patDataMemory::shuffleSample() {
  patShuffleVector<patIndividualData>()(&sample) ;
}

void patDataMemory::finalize() {
  
  for (vector<patIndividualData>::iterator ind = sample.begin() ;
       ind != sample.end() ;
       ++ind) {
    for (vector<patObservationData>::iterator i = ind->theObservations.begin() ;
	 i != ind->theObservations.end() ;
	 ++i) {
      observations.push_back(&(*i)) ;
    }
    for (vector<patAggregateObservationData>::iterator i = ind->theAggregateObservations.begin() ;
	 i != ind->theAggregateObservations.end() ;
	 ++i) {
      aggObservations.push_back(&(*i)) ;
    }
  }
}
