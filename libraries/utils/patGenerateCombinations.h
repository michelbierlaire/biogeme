//-*-c++-*------------------------------------------------------------
//
// File name : patGenerateCombinations.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec  7 10:08:14 2004
//
//--------------------------------------------------------------------

#ifndef patGenerateCombinations_h
#define patGenerateCombinations_h

#include <vector>
#include "patErrMiscError.h"
#include "patIterator.h"
#include "patDisplay.h"

/**
   @doc Given a family of variables with a finite number of values,
   generate all possible combinations of values. 
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Dec  7 10:08:14 2004)

 */


template <class InputIterator, class T> class patGenerateCombinations : 
public patIterator<vector<InputIterator>* > {

public:
  patGenerateCombinations(vector<InputIterator> bIterators, 
			  vector<InputIterator> eIterators,
			  patError*& err) : 
    beginIterators(bIterators), 
    endIterators(eIterators), 
    theIterators(bIterators),
    last(bIterators.size()-1),
    finish(patFALSE) {  
    if (bIterators.size() != eIterators.size()) {
      err = new patErrMiscError("Incompatible sizes") ;
      WARNING(err->describe()) ;
      return ;
    }

  } ;
  
  void first() {
    theIterators = beginIterators ;
    finish = patFALSE ;
  } ;

  void next() {
    //    DEBUG_MESSAGE("Increment counter " << last) ;
    ++theIterators[last] ;
    typename vector<InputIterator>::size_type currentIncrement = last ;
    while ((theIterators[currentIncrement] == 
	    endIterators[currentIncrement]) && 
	   !finish) {
      if (currentIncrement == 0) {
	finish = patTRUE ;
      }
      else {
	//	DEBUG_MESSAGE("Reset counter " << currentIncrement) ;
	theIterators[currentIncrement] = beginIterators[currentIncrement] ;
	--currentIncrement ;
	//	DEBUG_MESSAGE("Increment counter " << currentIncrement) ;
	++theIterators[currentIncrement] ;
      }
    }
  }

  patBoolean isDone() {
    if (theIterators.empty()) {
      return patTRUE ;
    }
    return (theIterators[0] == endIterators[0]) ;
  }

  vector<InputIterator>* currentItem() {
    return &theIterators ;
  }
  
  
  

private :
  vector<InputIterator> beginIterators ;
  vector<InputIterator> endIterators ;
  vector<InputIterator> theIterators ;
  typename vector<InputIterator>::size_type last ;
  patBoolean finish ;
};

#endif
