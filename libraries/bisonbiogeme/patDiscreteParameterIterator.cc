//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteParameterIterator.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Dec  5 18:18:39 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cassert>
#include "patDiscreteParameterIterator.h"
#include "patDiscreteParameter.h"

patDiscreteParameterIterator::patDiscreteParameterIterator(vector<patDiscreteParameter>* dataPtr) : theVector(dataPtr) {
  assert(dataPtr != NULL) ;
}

void patDiscreteParameterIterator::first() {
  i = theVector->begin() ;
}

void patDiscreteParameterIterator::next() {
  ++i ;
}


patBoolean patDiscreteParameterIterator::isDone() {
  return (i == theVector->end()) ;
}

patDiscreteParameter* patDiscreteParameterIterator::currentItem() {
  return (&(*i)) ;
}
