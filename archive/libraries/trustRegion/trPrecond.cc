//-*-c++-*------------------------------------------------------------
//
// File name : trPrecond.cc
// Author :    Michel Bierlaire
// Date :      Wed Jan 19 20:07:03 2000
//
//--------------------------------------------------------------------

#include <cmath>
#include <numeric>
#include "trPrecond.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

trVector trPrecond::solve(const trVector* b,
			  patError*& err) const {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }
  if (b == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return trVector() ;
  }
  return *b ;
}

