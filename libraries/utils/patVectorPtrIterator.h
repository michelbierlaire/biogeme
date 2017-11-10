//-*-c++-*------------------------------------------------------------
//
// File name : patVectorPtrIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Aug  6 16:17:44 2004
//
//--------------------------------------------------------------------

#ifndef patVectorPtrIterator_h
#define patVectorPtrIterator_h

/**
   @doc Generic  iterator on a STL vector containing pointers
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Fri Aug  6 16:17:44 2004)
 */

#include <vector>
#include "patIterator.h"

template <class T> class patVectorPtrIterator: public patIterator<T*> {

 public:
  patVectorPtrIterator<T>(vector< T* >* aVector) {
    theVector = aVector ;
  }
  ~patVectorPtrIterator<T>() {
    theVector = NULL ;
  }
  void first() {
    if (theVector != NULL) {
      iter = theVector->begin() ;
    }
  }
  void next() {
    ++iter ;
  }
  patBoolean isDone() {
    if (theVector == NULL) {
      return patTRUE ;
    }
    return (iter == theVector->end()) ;
  }
  T* currentItem() {
    return *iter ;
  }
 private:
  vector< T* >* theVector ;
  typename vector< T* >::iterator iter ;
  
};

#endif
