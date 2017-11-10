//-*-c++-*------------------------------------------------------------
//
// File name : patStlVectorIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Oct 31 17:08:25 2001
//
//--------------------------------------------------------------------

#ifndef patStlVectorIterator_h
#define patStlVectorIterator_h

#include <vector>
#include "patIterator.h"

/**
   @doc Iterator on a STL vector
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Oct 31 17:08:25 2001)
  */

template <class T>
class patStlVectorIterator : 
  public patIterator<T> {
public:
  /**
   */
  patStlVectorIterator(const vector<T>& v) : theVector(v) {}
  /**
   */
  void first() {
    i = theVector.begin() ;
  }
  /**
   */
  void next() {
    ++i ;
  }
  /**
   */
  patBoolean isDone() {
    return (i ==  theVector.end()) ;
  }
  /**
   */
  T currentItem() {
    return *i ;
  }
private:
  typename vector<T>::const_iterator i ;
  vector<T> theVector ;
};
#endif
