//-*-c++-*------------------------------------------------------------
//
// File name : patLargeNumbers.h
// Author :    Michel Bierlaire
// Date :      Mon Aug 31 21:50:57 2009
//
//--------------------------------------------------------------------


#ifndef patLargeNumbers_h
#define patLargeNumbers_h

#include "patType.h"

class patLargeNumbers {
  friend std::ostream& operator<<(std::ostream &str, const patLargeNumbers& x) ;
 public:
  patLargeNumbers(patULong n) ;
 private:
  patULong theNumber ;
  
};
#endif
