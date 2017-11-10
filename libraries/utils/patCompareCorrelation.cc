//-*-c++-*------------------------------------------------------------
//
// File name : patCompareCorrelation.cc
// Author :    Michel Bierlaire
// Date :      Wed Jul  4 16:17:54 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patCompareCorrelation.h"
#include "patConst.h" 
#include "patMath.h"

patBoolean patCompareCorrelation:: operator()(const patCorrelation& c1,
					      const patCorrelation& c2) const {

  if (c1.firstCoef == c2.firstCoef && c1.secondCoef == c2.secondCoef) {
    return patFALSE ;
  }
  if (patAbs(c1.robust_ttest) < patAbs(c2.robust_ttest)) {
    return patTRUE ;
  }
  else if (patAbs(c1.robust_ttest) > patAbs(c2.robust_ttest)) {
    return patFALSE ;
  }
  if (c1.firstCoef < c2.firstCoef) {
    return patTRUE ;
  }
  else if (c1.firstCoef > c2.firstCoef) {
    return patFALSE ;
  }
  if (c1.secondCoef < c2.secondCoef) {
    return patTRUE ;
  }
  return patFALSE ;
}
