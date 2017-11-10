//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileObsIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 14:50:24 2003
//
//--------------------------------------------------------------------

#ifndef patDataFileObsIterator_h
#define patDataFileObsIterator_h

#include "patIterator.h"
#include "patObservationData.h" 

class patDataFile ;
class patDataFileIndIterator ;
class patIndividualData ;

class patDataFileObsIterator: public patIterator<patObservationData*> {
 public:
  patDataFileObsIterator(patDataFile* aDataFile) ;
  ~patDataFileObsIterator() ;
  void first() ;
  void next() ;
  patBoolean isDone() ;
  patObservationData* currentItem() ;

 private:
  patDataFile* theDataFile ;
  patIterator<patIndividualData*>* theIndIterator ;
  vector<patObservationData>::iterator theObsIterator ;
  patIndividualData* currentIndividual ;
};

#endif
