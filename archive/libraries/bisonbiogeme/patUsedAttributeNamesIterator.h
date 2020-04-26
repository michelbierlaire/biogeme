//-*-c++-*------------------------------------------------------------
//
// File name : patUsedAttributeNamesIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sat Aug  2 12:30:39 2003
//
//--------------------------------------------------------------------

#ifndef patUsedAttributeNamesIterator_h
#define patUsedAttributeNamesIterator_h

#include <map>
#include "patString.h"
#include "patIterator.h"
#include "patRandomParameter.h"

class patArithRandom ;

/**
   @doc Iterator 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Sat Aug  2 12:30:39 2003)
 */
class patUsedAttributeNamesIterator: public patIterator<pair<patString,unsigned long> > {
 public:
  patUsedAttributeNamesIterator(map<patString,unsigned long>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  pair<patString,unsigned long> currentItem()  ;
 private:
  map<patString,unsigned long>* theMap ;
  map<patString,unsigned long>::const_iterator i ;
};
#endif 
