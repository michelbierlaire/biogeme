//-*-c++-*------------------------------------------------------------
//
// File name : patSequenceIterator.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 20:32:29 2001
//
//--------------------------------------------------------------------

#ifndef patSequenceIterator_h
#define patSequenceIterator_h

#include "patIterator.h"

/**
   @doc Implements a sequence a several iterator into one. 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Wed May 16 20:32:29 2001)
 */

template <class T> class patSequenceIterator : public patIterator<T> {
public:

  /**
   */
  patSequenceIterator(patIterator<T>* it = NULL) {
    addIterator(it) ;
  }

  /**
     Memory is released here
   */
  virtual ~patSequenceIterator() {
    releaseIterators() ;
  }
  
  /**
   */
  void addIterator(patIterator<T>* it) {
    if (it != NULL) {
      listOfIter.push_back(it) ;
    }
  }
  /**
   */
  void releaseIterators() {
    for (patULong i = 0 ;
	 i < listOfIter.size() ;
	 ++i) {
      DELETE_PTR(listOfIter[i]) ;
    }
  }
  /**
   */
  void first() {
    for (vecIter = listOfIter.begin() ;
	 vecIter != listOfIter.end() ;
	 ++vecIter) {
      (*vecIter)->first() ;
    }
    vecIter = listOfIter.begin() ;
    while (!isDone() && (*vecIter)->isDone()) {
      ++vecIter ;
    }
  }
  /**
   */
  void next() {
    (*vecIter)->next() ;
    while ((!isDone()) && (*vecIter)->isDone()) {
      ++vecIter ;
    }
  }
  /**
   */
  patBoolean isDone() {
    return (vecIter == listOfIter.end()) ;
  }
  /**
   */
  T currentItem() {
    if (isDone()) {
      WARNING("Should not call currentItem when iterator is done") ;
      return T() ;
    }
    else {
      return (*vecIter)->currentItem() ;
    }
  }

private:
  vector<patIterator<T>*> listOfIter ;
  typename vector<patIterator<T>*>::const_iterator vecIter ; 
};

#endif
