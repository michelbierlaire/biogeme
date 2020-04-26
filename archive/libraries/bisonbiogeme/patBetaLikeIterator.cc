//-*-c++-*------------------------------------------------------------
//
// File name : patBetaLikeIterator.cc
// Author :    Michel Bierlaire
// Date :      Wed May 16 16:08:31 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patBetaLikeIterator.h"

patBetaLikeIterator::patBetaLikeIterator(map<patString,patBetaLikeParameter>* x) : theMap(x) {
  first() ;
}
void patBetaLikeIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
void patBetaLikeIterator::next() {
  ++i ;
}

patBoolean patBetaLikeIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  else {
    return (i == theMap->end()) ;
  }
}
patBetaLikeParameter patBetaLikeIterator::currentItem() {
  if (!isDone()) {
    return i->second ;
  }
  else {
    return patBetaLikeParameter() ;
  }
}

