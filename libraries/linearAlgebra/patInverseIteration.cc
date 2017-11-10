//-*-c++-*------------------------------------------------------------
//
// File name : patInverseIteration.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Aug 14 14:30:20 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patInverseIteration.h"
#include "patConst.h"
#include "patMessage.h"
#include "patDisplay.h"
#include "patMath.h"

patInverseIteration::patInverseIteration(patMyMatrix* theMatrix) : 
  A(theMatrix),
  lu(theMatrix),
  z(theMatrix->nRows(),1.0),
  Az(12.0) {

}
  
patReal patInverseIteration::perturb(patReal mu,patError*& err) {

  patReal currMu = mu ;
  patBoolean invertible = patFALSE ;
  while (!invertible) {
    patMyMatrix LU(*A) ;
    for (unsigned long i = 0 ; i < LU.nRows() ; ++i) {
      LU[i][i] = LU[i][i] - currMu ;
    }
    lu.computeLu(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }

    if (!lu.isSuccessful()) {
      currMu += patEPSILON ;
    }
  }
  return currMu ;
}
  
void patInverseIteration::inverseIteration(unsigned long n, patError*& err) {

  patBoolean stop = patFALSE ;
  unsigned long iter = 0 ;
  while(!stop) {
    ++iter ;
    // Normalize z ;
    patVariables normZ(z) ;
    patReal norm = 0.0 ;
    for (unsigned long i = 0 ; i < normZ.size() ; ++i) {
      norm += normZ[i] * normZ[i] ;
    }
    norm = sqrt(norm) ;
    for (unsigned long i = 0 ; i < normZ.size() ; ++i) {
      normZ[i] /= norm ;
    }

    // Solve
    z = *lu.solve(normZ,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    // Check if we continue

    if (iter >= n) {
      stop = patTRUE ;
    } 
    else {
      patVariables isZero(A->nRows()) ;
      multVec(*A,z,isZero,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal zero = 0.0 ;
      for (unsigned long i = 0 ; i < isZero.size() ; ++i) {
	zero += isZero[i] * isZero[i] ;
      }
      Az = sqrt(zero) ;
      stop =  (Az <= patEPSILON * patEPSILON)  ;
    }
  }
}

patVariables patInverseIteration::getEigenVector() {
  patVariables result(z.size()) ;
  patReal first = 0.0 ;
  for (unsigned long i = 0 ; i < z.size() ; ++i) {
    if (patAbs(z[i]) < patEPSILON * patEPSILON) {
      result[i] = 0.0 ;
    }
    else {
      if (first == 0.0) {
	first = z[i] ;
      }
      result[i] = z[i] / first ; 
    }
  }
  return result ;
}

patReal patInverseIteration::getAz(patError*& err) {
  patVariables isZero(A->nRows()) ;
  multVec(*A,z,isZero,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  patReal zero = 0.0 ;
  for (unsigned long i = 0 ; i < isZero.size() ; ++i) {
    zero += isZero[i] * isZero[i] ;
  }
  Az = sqrt(zero) ;
  return Az ;
}
