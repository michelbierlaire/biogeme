//-*-c++-*------------------------------------------------------------
//
// File name : patNrSgn.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Aug 15 15:39:14 2005
//
//--------------------------------------------------------------------

#ifndef patNrSgn_h
#define patNrSgn_h

#include "patType.h"

inline patReal patNrSgn(const patReal& a, const patReal& b) {
  return (b >= 0) ? ((a >= 0) ? a : -a) : ((a >= 0) ? -a : a) ;
}

#endif
