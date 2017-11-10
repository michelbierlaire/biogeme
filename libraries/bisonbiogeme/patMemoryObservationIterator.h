//-*-c++-*------------------------------------------------------------
//
// File name : patMemoryObservationIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Mar 29 22:42:33 2004
//
//--------------------------------------------------------------------

#ifndef patMemoryObservationIterator_h
#define patMemoryObservationIterator_h

#include "patIterator.h"
#include "patObservationData.h"
#include "patIndividualData.h"

class patMemoryObservationIterator : 
  public patIterator<patObservationData*> {
public:
  /**
   */
  patMemoryObservationIterator(vector<patIndividualData>* theData) ;
  
  /**
   */
  void first() ;
  /**
   */
  void next() ;
  /**
   */
  patBoolean isDone() ;
  /**
   */
  patObservationData* currentItem() ;

private:
  vector<patIndividualData>* theData ;
  vector<patIndividualData>::iterator individualIter ;
  vector<patObservationData>::iterator obsIter ;
};

#endif
