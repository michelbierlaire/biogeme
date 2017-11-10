//-*-c++-*------------------------------------------------------------
//
// File name : patMath.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Feb  4 16:07:10 1999
//
//--------------------------------------------------------------------

#ifndef patMath_h
#define patMath_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cmath>

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

/**
 */
// template <class T> T patAbs(T x) {
//   return ((x<0)?-x:x) ;
// }

#define patAbs(xx) ((xx<0)?-(xx):(xx))

#define patFinite(xx) (isfinite(xx))

#define patExp(xx) ((!patFinite(xx) || (xx) >= patLogMaxReal::the())?patMaxReal:exp(xx))

#ifdef HAVE_CEIL
#define patCeil(xx) (ceil(xx))
#endif

/**
 */
template <class T> T patMin(T a, T b) {
  return((a < b)?a:b) ;
}

/**
 */
template <class T> T patMax(T a, T b) {
  return((b < a)?a : b) ;
}

/**
 */
template <class T> T patSgn(T a) {
  return((a < 0)?T(-1):T(1)) ;
}

#endif
