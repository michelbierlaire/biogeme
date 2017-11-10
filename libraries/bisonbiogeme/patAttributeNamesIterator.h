//-*-c++-*------------------------------------------------------------
//
// File name : patAttributeNamesIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Mar  5 14:55:44 2003
//
//--------------------------------------------------------------------

#ifndef patAttributeNamesIterator_h
#define patAttributeNamesIterator_h

#include <map>
#include "patString.h"
#include "patIterator.h"
#include "patRandomParameter.h"

class patArithRandom ;

/**
   @doc Iterator on a map<patString,pair<patRandomParameter,patArithRandom*> > generating the names of draws
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Mar  5 14:55:44 2003)
 */
class patAttributeNamesIterator: public patIterator<patString> {
 public:
  patAttributeNamesIterator(vector<patString>* x) ;
  void first()  ;
  void next()  ;
  patBoolean isDone()  ;
  patString currentItem()  ;
 private:
  vector<patString>* theMap ;
  vector<patString>::const_iterator i ;
};
#endif 
