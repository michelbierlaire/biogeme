//-*-c++-*------------------------------------------------------------
//
// File name : bioParameterIterator.h
// Author :    Michel Bierlaire
// Date :      Mon Apr 25 20:24:20 2016
//
//--------------------------------------------------------------------

#ifndef bioParameterIterator_h
#define bioParameterIterator_h

#include <map>
#include "patIterator.h"
#include "bioOneParameter.h"

template <class T> class bioParameterIterator: public patIterator<bioOneParameter<T> > {
 public:
  bioParameterIterator<T>(map<patString,pair<T,patString> >* v) :
  theValues(v),
    iter(v->begin()) {
  }
  
  void first() {
    iter = theValues->begin() ;
  }

  void next() {
    ++iter;
  }
  patBoolean isDone() {
    return (iter == theValues->end()) ;
  }
  
  bioOneParameter<T> currentItem()  {
    if (!isDone()) {
      return bioOneParameter<T>(iter->first,iter->second.first,iter->second.second) ;
    }
    return bioOneParameter<T>() ;
  }
  
 private:
  map<patString,pair<T,patString> >* theValues ;
  typename map<patString,pair<T,patString> >::iterator iter ;
  
} ;
#endif
