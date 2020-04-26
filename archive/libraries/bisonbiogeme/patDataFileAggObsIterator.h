//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileAggObsIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Jan 15 11:59:10 2006
//
//--------------------------------------------------------------------

#ifndef patDataFileAggObsIterator_h
#define patDataFileAggObsIterator_h

#include "patIterator.h"
#include "patObservationData.h" 

class patDataFile ;
class patDataFileIndIterator ;
class patIndividualData ;

class patDataFileAggObsIterator: public patIterator<patAggregateObservationData*> {
 public:
  patDataFileAggObsIterator(patDataFile* aDataFile) ;
  ~patDataFileAggObsIterator() ;
  void first() ;
  void next() ;
  patBoolean isDone() ;
  patAggregateObservationData* currentItem() ;

 private:
  patDataFile* theDataFile ;
  patIterator<patIndividualData*>* theIndIterator ;
  vector<patAggregateObservationData>::iterator theAggObsIterator ;
  patIndividualData* currentIndividual ;
};

#endif
