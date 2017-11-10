//-*-c++-*------------------------------------------------------------
//
// File name : patInverse.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:11:28 2005
//
//--------------------------------------------------------------------


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patInverse.h"
#include "patMyMatrix.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patInverse::patInverse(patMyMatrix* aMat) : lu(aMat),
					    theMatrix(aMat) {
  if (aMat == NULL) {
    return ;
  }
  n = theMatrix->nRows() ;
  solution = new patMyMatrix(n,n) ;
}

patInverse::~patInverse() {
  DELETE_PTR(solution) ;
}

const patMyMatrix* patInverse::computeInverse(patError*& err) {
  if (theMatrix == NULL) {
    err = new patErrNullPointer("patMatrix") ;
    WARNING(err->describe()) ;
    return NULL;
  }
  unsigned long i ;
  unsigned long j ;
  
  patVariables col(n) ;

  for (j = 0 ; j < n ; ++j) {
    for (i = 0 ; i < n ; ++i) {
      col[i] = 0.0 ;
    }
    col[j] = 1.0 ;
    const patVariables* sol = lu.solve(col,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    for (i = 0 ; i < n ; ++i) {
      (*solution)[i][j] = (*sol)[i] ;
    }
  }
  return solution ;
}


patBoolean patInverse::isInvertible() const {
  return lu.isSuccessful() ;
}
