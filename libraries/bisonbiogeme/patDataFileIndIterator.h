//-*-c++-*------------------------------------------------------------
//
// File name : patDataFileIndIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Mar 30 07:44:47 2004
//
//--------------------------------------------------------------------

#ifndef patDataFileIndIterator_h
#define patDataFileIndIterator_h

#include "patIterator.h"
#include "patIndividualData.h" 

class patDataFileIndIterator: public patIterator<patIndividualData*> {
 public:
  patDataFileIndIterator(patString aFile,patIndividualData aRow) ;
  ~patDataFileIndIterator() ;
  void first() ;
  void next() ;
  patBoolean isDone() ;
  patIndividualData* currentItem() ;

 private:
  void read() ;
 private:
  patString fileName ;
  ifstream inFile ;
  patIndividualData currentIndividual ;
  patBoolean error ;
  
  unsigned long counter ;

};

#endif
