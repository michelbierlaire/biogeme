//-*-c++-*------------------------------------------------------------
//
// File name : patUsedAttributeNamesIterator.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sat Aug  2 12:27:42 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patUsedAttributeNamesIterator.h"

patUsedAttributeNamesIterator::patUsedAttributeNamesIterator(map<patString,unsigned long>* x) : theMap(x) {

}
void patUsedAttributeNamesIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
  
void patUsedAttributeNamesIterator::next() {
  ++i ;
}

patBoolean patUsedAttributeNamesIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  return (i == theMap->end()) ;
}

pair<patString,unsigned long> patUsedAttributeNamesIterator::currentItem() {
  if (theMap == NULL) {
    return pair<patString,unsigned long>() ;
  }
  return (*i) ;
}
