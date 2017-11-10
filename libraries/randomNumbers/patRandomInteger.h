//-*-c++-*------------------------------------------------------------
//
// File name : patRandomInteger.h
// Author :    Michel Bierlaire
// Date :      Thu Aug 12 09:45:40 2010
//
//--------------------------------------------------------------------

#ifndef patRandomInteger_h
#define patRandomInteger_h

#include <stdlib.h>
#include "patType.h"

class patRandomInteger {
 public:
  patRandomInteger(patULong seed = 9021967) ;
  // Draw a random integer between minNumber and maxNumber-1, with equal probability
  patULong drawNumber(patULong minNumber, patULong maxNumber) ;

};
#endif
