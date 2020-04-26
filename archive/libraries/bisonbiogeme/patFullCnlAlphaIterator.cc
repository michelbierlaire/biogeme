//-*-c++-*------------------------------------------------------------
//
// File name : patFullCnlAlphaIterator.cc
// Author :    Michel Bierlaire
// Date :      Sun May 20 20:35:55 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patFullCnlAlphaIterator.h"

patFullCnlAlphaIterator::patFullCnlAlphaIterator(map<patString,patCnlAlphaParameter>* x) : theMap(x) {
  first() ;
}
void patFullCnlAlphaIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
void patFullCnlAlphaIterator::next() {
  ++i ;
}

patBoolean patFullCnlAlphaIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  else {
    return (i == theMap->end()) ;
  }
}
patCnlAlphaParameter patFullCnlAlphaIterator::currentItem() {
  if (!isDone()) {
    return i->second ;
  }
  else {
    return patCnlAlphaParameter() ;
  }
}

