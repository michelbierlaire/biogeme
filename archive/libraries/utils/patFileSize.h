//-*-c++-*------------------------------------------------------------
//
// File name : patFileSize.h
// Author :    Michel Bierlaire
// Date :      Mon Dec 19 08:59:01 2011
//
//--------------------------------------------------------------------

// Class to display the size of a file in human readable format

#ifndef patFileSize_h
#define patFileSize_h

#include "patType.h"

class patFileSize {
  friend std::ostream& operator<<(std::ostream &str, const patFileSize& x) ;
 public:
  patFileSize(patULong n) ;
 private:
  patULong theNumber ;
  
};
#endif
