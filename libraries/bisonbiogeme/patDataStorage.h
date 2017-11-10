//-*-c++-*------------------------------------------------------------
//
// File name : patDataStorage.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Dec  5 10:50:53 2003
//
// Virtual class to define the interface for the data storage
//
//--------------------------------------------------------------------

#ifndef patDataStorage_h
#define patDataStorage_h

#include "patConst.h"
#include "patIterator.h"
#include "patError.h"

class patIndividualData ;
class patObservationData ;
class patAggregateObservationData ;

class patDataStorage {

 public:
  /**
   */
  virtual ~patDataStorage() {};
  /**
     Add a data at the end 
   */
  virtual void push_back(patIndividualData* data) = PURE_VIRTUAL;
  /**
   */
  virtual unsigned long getSize() = PURE_VIRTUAL ;
  /**
     Create an iterator. It is the responsibility of the caller to
     release the memory
   */
  virtual patIterator<patIndividualData*>* createIndIterator() = PURE_VIRTUAL ;

  /**
   */
  virtual void releaseIndIterator() = PURE_VIRTUAL ;

  /**
     Create an iterator. It is the responsibility of the caller to
     release the memory
   */
  virtual patIterator<patObservationData*>* createObsIterator() = PURE_VIRTUAL ;
  /**
   */
  virtual void releaseObsIterator() = PURE_VIRTUAL ;
  
  /**
     Create a vector of iterators, each one spanning a
     different portion of the data, so that they can be associated with
     different threads. It is the responsibility of the caller to
     release the memory
   */
  virtual vector<patIterator<patObservationData*>*> 
  createObsIteratorThread(unsigned int nbrThreads, 
			  patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual void releaseObsIteratorThread() = PURE_VIRTUAL ;

  /**
     Create a vector of iterators, each one spanning a
     different portion of the data, so that they can be associated with
     different threads. It is the responsibility of the caller to
     release the memory
   */
  virtual vector<patIterator<patIndividualData*>*> 
  createIndIteratorThread(unsigned int nbrThreads, 
			  patError*& err) = PURE_VIRTUAL ;

  /**
   */
  virtual void releaseIndIteratorThread() = PURE_VIRTUAL ;

  /**
     Create an iterator. It is the responsibility of the caller to
     release the memory
   */
  virtual patIterator<patAggregateObservationData*>* createAggObsIterator() = PURE_VIRTUAL ;

  /**
   */
  virtual void releaseAggObsIterator() = PURE_VIRTUAL ;

  /**
     Should be called after data have been written
   */
  virtual void finalize() = PURE_VIRTUAL ;
  /**
     Empty the database
   */
  virtual void erase() = PURE_VIRTUAL ;

  /**
     For storage in memory, it allows to reserve a priori the required space. 
   */
  virtual void reserveMemory(unsigned long size, 
			     patIndividualData* instance) = PURE_VIRTUAL ;
  
};

#endif
