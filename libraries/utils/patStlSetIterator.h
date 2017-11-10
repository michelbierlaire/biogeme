//-*-c++-*------------------------------------------------------------
//
// File name : patStlSetIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue May 19 17:19:59 2009
//
//--------------------------------------------------------------------

#ifndef patStlSetIterator_h
#define patStlSetIterator_h

#include <vector>
#include "patIterator.h"

/**
   @doc Iterator on a STL set
  */

template <class T>
class patStlSetIterator : 
  public patIterator<T> {
public:
  /**
   */
  patStlSetIterator(const set<T>& v) : theSet(v) {}
  /**
   */
  void first() {
    i = theSet.begin() ;
  }
  /**
   */
  void next() {
    ++i ;
  }
  /**
   */
  patBoolean isDone() {
    return (i ==  theSet.end()) ;
  }
  /**
   */
  T currentItem() {
    return *i ;
  }
private:
  typename set<T>::const_iterator i ;
  set<T> theSet ;
};
#endif
