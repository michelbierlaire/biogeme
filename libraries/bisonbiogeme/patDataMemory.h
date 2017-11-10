//-*-c++-*------------------------------------------------------------
//
// File name : patDataMemory.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 10:50:59 2003
//
// Class storing the data in memory.
//
//--------------------------------------------------------------------

#ifndef patDataMemory_h
#define patDataMemory_h

#include "patDataStorage.h"

class patDataMemory : public patDataStorage {

 public:
  /**
   */
  patDataMemory() ;
  /**
   */
  ~patDataMemory() ;
  /**
     Add a data at the end 
   */
  void push_back(patIndividualData* data) ;
  /**
   */
  unsigned long getSize()  ;
  /**
     Empty the database
   */
  void erase()  ;
  /**
     Create an iterator. It is the responsibility of the caller to
     relase the meory
   */
  patIterator<patIndividualData*>* createIndIterator() ;

  void releaseIndIterator() ;
  /**
     Create an iterator. It is the responsibility of the caller to
     release the memory
   */
  patIterator<patObservationData*>* createObsIterator()  ;
  /**
   */
  void releaseObsIterator() ;

  /**
     Create a vector of iterators, each one spanning a
     different portion of the data, so that they can be associated with
     different threads. It is the responsibility of the caller to
     release the memory
   */
  vector<patIterator<patObservationData*>*> 
  createObsIteratorThread(unsigned int nbrThreads, 
			  patError*& err)  ;
 void releaseObsIteratorThread() ;
  
  /**
     Create a vector of iterators, each one spanning a
     different portion of the data, so that they can be associated with
     different threads. It is the responsibility of the caller to
     release the memory
   */
  vector<patIterator<patIndividualData*>*> 
  createIndIteratorThread(unsigned int nbrThreads, 
			  patError*& err)  ;
  void releaseIndIteratorThread() ;
  
  /**
     Create an iterator. It is the responsibility of the caller to
     release the memory
   */
  patIterator<patAggregateObservationData*>* createAggObsIterator() ;
  void releaseAggObsIterator()  ;

  /**
     For storage in memory, it allows to reserve a priori the required space. 
   */
  void reserveMemory(unsigned long size, patIndividualData* instance)  ;

  /**
   */
  void shuffleSample() ;

  /**
     Should be called after data have been written
   */
  void finalize() ;

 protected:

  vector<patObservationData*> observations ;
  vector<patAggregateObservationData*> aggObservations ;
  vector<patIndividualData> sample ;
  //  vector<patIndividualData>::iterator iter ;

  patIterator<patIndividualData*>* indIter ;
  patIterator<patObservationData*>* obsIter ;
  vector<patIterator<patObservationData*>*> obsIterThread ;
  vector<patIterator<patIndividualData*>*>  indIterThread ;
  patIterator<patAggregateObservationData*>* aggObsIter ; 


};

#endif
