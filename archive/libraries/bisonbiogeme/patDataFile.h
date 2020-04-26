//-*-c++-*------------------------------------------------------------
//
// File name : patDataFile.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 14:17:54 2003
//
// Class storing the data in a binary file.
//
//--------------------------------------------------------------------

#ifndef patDataFile_h
#define patDataFile_h

#include "patDataStorage.h"
#include "patIndividualData.h"
#include "patObservationData.h"

class patDataFile : public patDataStorage {

 public:
  /**
   */
  patDataFile() ;
  /**
   */
  ~patDataFile() ;
  /**
     Add a data at the end 
   */
  void push_back(patIndividualData* data) ;
  /**
     Add panel draws
   */
  void push_panel_draws_back(vector<patVariables>* draws) ;

  /**
   */
  unsigned long getSize()  ;

  /**
   */
  unsigned long getPanelSize()  ;

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
  patIterator<patObservationData*>* createObsIterator() ;

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

  void releaseAggObsIterator() ;
  /**
     For storage in memory, it allows to reserve a priori the required space. 
   */
  void reserveMemory(unsigned long size, patIndividualData* instance)  ;
  
  /**
     Should be called after data have been written
   */
  void finalize() ;
  
  /**
     For storage in memory, reserve the space to stock draws for panel data
   */
  void reservePanelMemory(unsigned long size, 
			  vector<patVariables>* instance) ;


private :
  patIndividualData typicalIndividual ;
  patString fileName ;
  patString panelFileName ;
  ofstream outFile ;
  unsigned long size ;

  patIterator<patObservationData*>* obsIter ;
  patIterator<patAggregateObservationData*>* aggObsIter ;
  patIterator<patIndividualData*>* indIter ;
};

#endif
