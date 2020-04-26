//-*-c++-*------------------------------------------------------------
//
// File name : patShuffleVector.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Apr  4 15:39:30 2004
//
//--------------------------------------------------------------------

#ifndef patShuffleVector_h
#define patShuffleVector_h


/**
@doc Replace the random_shuffle STL function that does not work
@author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Sun Apr  4 15:39:30 2004)
*/

#include <stdlib.h>
#include <vector>
#include <cassert>

template <class T> class patShuffleVector {

 public:
  /**
   */
  void operator()(vector< T >* theVector) {
    assert(theVector != NULL) ;
    for (patULong i=0; i<theVector->size(); ++i) {
      int r = rand() % theVector->size(); 
      T temp = (*theVector)[i]; 
      (*theVector)[i] = (*theVector)[r];
      (*theVector)[r] = temp;
    }
  }
};
#endif
