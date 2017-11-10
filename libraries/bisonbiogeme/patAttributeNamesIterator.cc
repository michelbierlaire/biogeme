//-*-c++-*------------------------------------------------------------
//
// File name : patAttributeNamesIterator.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Mar  4 21:11:51 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patAttributeNamesIterator.h"

patAttributeNamesIterator::patAttributeNamesIterator(vector<patString>* x) : theMap(x) {

}
void patAttributeNamesIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
  
void patAttributeNamesIterator::next() {
  ++i ;
}

patBoolean patAttributeNamesIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  return (i == theMap->end()) ;
}

patString patAttributeNamesIterator::currentItem() {
  if (theMap == NULL) {
    return patString() ;
  }
  return (*i) ;
}
