//-*-c++-*------------------------------------------------------------
//
// File name : patLoopTime.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Sep  4 12:25:24 2005
//
//--------------------------------------------------------------------

#ifndef patLoopTime_h
#define patLoopTime_h

#include "patAbsTime.h"

/**
   @doc Estimate the time when a loop will finish.
 */

class patLoopTime {

  friend ostream& operator<<(ostream& stream, const patLoopTime& t) ;

 public:
  patLoopTime(unsigned long numberOfIterations) ;
  void setIteration(unsigned long iter) ;

 private:
  patAbsTime startingTime ;
  unsigned long currentIter ;
  patAbsTime currentTime ;
  unsigned long numberOfIterations ;
  patAbsTime terminationTime ;
};


#endif
