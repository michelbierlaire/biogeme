//-*-c++-*------------------------------------------------------------
//
// File name : patConstraintNestIterator.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Sep 27 16:55:04 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patConstraintNestIterator.h"

patConstraintNestIterator::patConstraintNestIterator(const vector<pair<unsigned long, unsigned long> >& v) : theVector(v) {

}
void patConstraintNestIterator::first() {
  i = theVector.begin() ;
}
void patConstraintNestIterator::next() {
  ++i ;
}
patBoolean patConstraintNestIterator::isDone() {
  return (i == theVector.end()) ;
}
pair<unsigned long, unsigned long> patConstraintNestIterator::currentItem() {
  return *i ;
}
