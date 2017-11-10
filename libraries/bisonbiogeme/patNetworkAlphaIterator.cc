//-*-c++-*------------------------------------------------------------
//
// File name : patNetworkAlphaIterator.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Dec 16 21:40:35 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNetworkAlphaIterator.h"

patNetworkAlphaIterator::patNetworkAlphaIterator(map<patString,patNetworkGevLinkParameter>* x) : theMap(x) {

}
void patNetworkAlphaIterator::first() {
  i = theMap->begin() ;
}
void patNetworkAlphaIterator::next() {
  ++i ;
}
patBoolean patNetworkAlphaIterator::isDone() {
  return (i == theMap->end()) ;
}
patBetaLikeParameter patNetworkAlphaIterator::currentItem() {
  return i->second.alpha ;
}
