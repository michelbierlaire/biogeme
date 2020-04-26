//-*-c++-*------------------------------------------------------------
//
// File name : patNlNestIterator.cc
// Author :    Michel Bierlaire
// Date :      Wed May 16 16:08:31 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNlNestIterator.h"

patNlNestIterator::patNlNestIterator(map<patString,patNlNestDefinition>* x) : theMap(x) {
  first() ;
}
void patNlNestIterator::first() {
  if (theMap != NULL) {
    i = theMap->begin() ;
  }
}
void patNlNestIterator::next() {
  ++i ;
}

patBoolean patNlNestIterator::isDone() {
  if (theMap == NULL) {
    return patTRUE ;
  }
  else {
    return (i == theMap->end()) ;
  }
}
patBetaLikeParameter patNlNestIterator::currentItem() {
  if (!isDone()) {
    return i->second.nestCoef ;
  }
  else {
    return patBetaLikeParameter() ;
  }
}

