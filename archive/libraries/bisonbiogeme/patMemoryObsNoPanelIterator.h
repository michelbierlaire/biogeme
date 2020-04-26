//-*-c++-*------------------------------------------------------------
//
// File name : patMemoryObsNoPanelIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Aug  3 18:14:12 2004
//
//--------------------------------------------------------------------

#ifndef patMemoryObsNoPanelIterator_h
#define patMemoryObsNoPanelIterator_h

#include "patIterator.h"
#include "patObservationData.h"
#include "patIndividualData.h"

/**
   This iterator assumes that there is exactly one observation per individual
 */

class patMemoryObsNoPanelIterator : 

  public patIterator<patObservationData*> {
public:
  /**
   */
  patMemoryObsNoPanelIterator(vector<patIndividualData>* theData) ;
  
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
};

#endif
