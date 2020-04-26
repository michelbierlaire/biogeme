//-*-c++-*------------------------------------------------------------
//
// File name : patLoop.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Aug  1 11:58:45 2003
//
//--------------------------------------------------------------------

#ifndef patLoop_h
#define patLoop_h

#include "patString.h"

struct patLoop {

  patString variable ;
  long lower ;
  long upper ;
  long step ;
};

#endif
