//-*-c++-*------------------------------------------------------------
//
// File name : patSubvectorIterator.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Mon Oct 15 09:34:01 2007
//
//--------------------------------------------------------------------

#ifndef patSubvectorIterator_h
#define patSubvectorIterator_h

/**
   @doc Iterator on a portion of a STL vector, returning pointers to
   the data 
 */

#include <vector>
#include "patIterator.h"

template <class T> class patSubvectorIterator: public patIterator<T*> {

 public:
  // Iterator start at element number b, and stops at element number e-1
  patSubvectorIterator<T>(vector< T >* aVector, 
			  patULong b, 
			  patULong e) {
    theVector = aVector ;
    start = b ;
    stop = e ;
  }
  void first() {
    if (theVector != NULL) {
      index = start ;
    }
  }
  void next() {
    ++index ;
  }
  patBoolean isDone() {
    if (theVector == NULL) {
      return patTRUE ;
    }
    return (index == stop || index == theVector->size()) ;
  }
  T* currentItem() {
    return &((*theVector)[index]) ;
  }
 private:
  vector< T >* theVector ;
  typename vector< T >::size_type index ;
  typename vector< T >::size_type start ;
  typename vector< T >::size_type stop ;
  
};

#endif
