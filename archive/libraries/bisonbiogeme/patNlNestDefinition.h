//-*-c++-*------------------------------------------------------------
//
// File name : patNlNestDefinition.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 16:29:55 2001
//
//--------------------------------------------------------------------


#ifndef patNlNestDefinition_h
#define patNlNestDefinition_h

#include <list>
#include "patBetaLikeParameter.h"

struct patNlNestDefinition {
  /**
   */
  patBetaLikeParameter nestCoef ;
  /**
   */
  list<long> altInNest ;
};


#endif
