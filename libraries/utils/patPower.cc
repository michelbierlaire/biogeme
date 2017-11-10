//-*-c++-*------------------------------------------------------------
//
// File name : patPower.cc
// Author :    Michel Bierlaire
// Date :      Thu May 24 13:53:48 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patDisplay.h"
#include "patMath.h"
#include "patPower.h"

patReal patPower(patReal x, patReal y) {
  if (patAbs(x) < patEPSILON) {
    return 0.0 ;
  }
  patReal result = pow(x,y) ;
  if (patFinite(result) == 0) {
    //    WARNING("Unable to compute " << x << " to the power " << y) ;
    return 0.0 ;
  }
  return result ;
}
