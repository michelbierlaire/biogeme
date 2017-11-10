//-*-c++-*------------------------------------------------------------
//
// File name : patInverse.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:08:22 2005
//
//--------------------------------------------------------------------

#ifndef patInverse_h
#define patInverse_h

#include "patError.h"
#include "patLu.h"

class patMyMatrix ;

class patInverse {

 public:
  patInverse(patMyMatrix* aMat) ;
  const patMyMatrix* computeInverse(patError*& err) ;
  patBoolean isInvertible() const ;
  ~patInverse() ;

 private:
  unsigned long n ;
  patLu lu ;
  patMyMatrix* theMatrix ;
  patMyMatrix* solution ;
};


#endif
