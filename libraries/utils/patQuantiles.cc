//-*-c++-*------------------------------------------------------------
//
// File name : patQuantiles.cc
// Author :   Michel Bierlaire
// Date :     Sun May  6 08:29:30 2012
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <algorithm>
#include "patDisplay.h"
#include "patQuantiles.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"

patQuantiles::patQuantiles(patVariables* x) : data(x) {
  if (data != NULL) {
    sort(data->begin(),data->end()) ;
  }
}

patReal patQuantiles::getQuantile(patReal p, patError*& err) {
  if (data == NULL) {
    err = new patErrNullPointer("patVariables") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (data->size() == 0) {
    err = new patErrMiscError("Empty database") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (p < 0 || p > 1) {
    err = new patErrOutOfRange<patReal>(p,0,1) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (p == 0.0) {
    return *data->begin() ;
  }
  if (p == 1.0) {
    return *data->rbegin() ;
  }
  patULong n = data->size() ;
  
  patReal np = patReal(n) * p + 0.5 ;
  patULong up = floor(np+0.5) ;
  patULong down = ceil(np-0.5) ;
  patReal q = 0.5 * ((*data)[up-1] + (*data)[down-1]) ;
  return q ;
}

