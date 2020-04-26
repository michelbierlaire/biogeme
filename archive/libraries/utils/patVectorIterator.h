//-*-c++-*------------------------------------------------------------
//
// File name : patVectorIterator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Dec  5 11:30:42 2003
//
//--------------------------------------------------------------------

#ifndef patVectorIterator_h
#define patVectorIterator_h

/**
   @doc Generic  iterator on a STL vector, returning pointers to the data
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Fri Dec  5 11:30:42 2003)
 */

#include <vector>
#include "patIterator.h"

template <class T> class patVectorIterator: public patIterator<T*> {

 public:
  patVectorIterator<T>(vector< T >* aVector) {
    theVector = aVector ;
  }
  virtual ~patVectorIterator() {
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
    return &(*iter) ;
  }
 private:
  vector< T >* theVector ;
  typename vector< T >::iterator iter ;
  
};

#endif
