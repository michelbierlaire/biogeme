//-*-c++-*------------------------------------------------------------
//
// File name : patCnlAlphaIterator.cc
// Author :    Michel Bierlaire
// Date :      Wed May 16 17:36:12 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patCnlAlphaIterator.h"

patCnlAlphaIterator::patCnlAlphaIterator(map<patString,patCnlAlphaParameter>* x) : theMap(x) {
  first() ;
}
void patCnlAlphaIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
void patCnlAlphaIterator::next() {
  ++i ;
}

patBoolean patCnlAlphaIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  else {
    return (i == theMap->end()) ;
  }
}
patBetaLikeParameter patCnlAlphaIterator::currentItem() {
  if (!isDone()) {
    return i->second.alpha ;
  }
  else {
    return patBetaLikeParameter() ;
  }
}

