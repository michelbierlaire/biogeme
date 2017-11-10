//-*-c++-*------------------------------------------------------------
//
// File name : patPValue.cc
// Author :    Michel Bierlaire
// Date :      Wed Jun 14 22:01:32 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patPValue.h"
#include "patDisplay.h"
#include "patNormalCdf.h"

patReal patPValue(patReal test, patError*& err) {

  if (test > 5.9) {
    return 0.0 ;
  }

  patReal result = 2.0*(1.0 - patNormalCdf::the()->compute(test,err)) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return result ;

}
