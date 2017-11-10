//-*-c++-*------------------------------------------------------------
//
// File name : patPythag.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Aug 15 17:23:48 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patPythag.h"
#include "patMath.h"


patReal patPythag(const patReal a, const patReal b) {
  patReal absa, absb ;
  absa = patAbs(a) ;
  absb = patAbs(b) ;
  if (absa > absb) {
    patReal frac = absb / absa ; 
    return absa * sqrt(1.0 + frac * frac) ;
  }
  else {
    if (absb == 0.0) {
      return 0.0 ;
    }
    patReal frac = absa / absb ;
    return absb * sqrt(1.0 + frac * frac) ;
  }
}
