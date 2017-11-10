//-*-c++-*------------------------------------------------------------
//
// File name : patThreeStrings.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed Oct 26 14:57:10 2005
//
//--------------------------------------------------------------------

#ifndef patThreeStrings_h
#define patThreeStrings_h

#include "patString.h"

struct patThreeStrings {  
  patThreeStrings(patString a, patString b, patString c) :
    s1(a),s2(b),s3(c) {} ;
  patString s1 ;  
  patString s2 ; 
  patString s3 ;
} ;


#endif
