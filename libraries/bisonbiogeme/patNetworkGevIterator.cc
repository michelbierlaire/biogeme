//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkGevIterator.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Sep 11 14:57:33 2002
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNetworkGevIterator.h"

patNetworkGevIterator::patNetworkGevIterator(vector<patNetworkGevNode*>* aList) : theList(aList) {

}

void patNetworkGevIterator::first() {
  if (theList != NULL) {
    theIterator = theList->begin() ;
  }
}
void patNetworkGevIterator::next() {
  if (theList != NULL) {
    ++theIterator ;
  }
}
patBoolean patNetworkGevIterator::isDone() {
  if (theList != NULL) {
    return (theIterator == theList->end()) ;
  }
  else {
    return patTRUE ;
  }
}

patNetworkGevNode* patNetworkGevIterator::currentItem() {
  if (theList != NULL) {
    return *theIterator ;
  }
  else {
    return NULL ;
  }
}

