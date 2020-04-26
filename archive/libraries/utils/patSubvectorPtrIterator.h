//-*-c++-*------------------------------------------------------------
//
// File name : patSubvectorPtrIterator.h
// Author :    Michel Bierlaire
// Date :      Mon Oct 15 10:05:18 2007
//
//--------------------------------------------------------------------

#ifndef patSubvectorPtrIterator_h
#define patSubvectorPtrIterator_h

/**
   @doc Generic  iterator on a STL vector containing pointers
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Fri Aug  6 16:17:44 2004)
 */

#include <vector>
#include "patIterator.h"

template <class T> class patSubvectorPtrIterator: public patIterator<T*> {

 public:
  patSubvectorPtrIterator<T>(vector< T* >* aVector, 
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
    return (*theVector)[index] ;
  }
 private:
  vector< T* >* theVector ;
  typename vector< T* >::size_type index ;
  typename vector< T >::size_type start ;
  typename vector< T >::size_type stop ;
  
};

#endif
