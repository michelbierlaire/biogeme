//-*-c++-*------------------------------------------------------------
//
// File name : bioGaussHermite.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 10:57:03 2010
//
//-------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bioGaussHermite.h"
#include "patDisplay.h"
#include "patErrNullPointer.h"

bioGaussHermite::bioGaussHermite(bioGhFunction* f) : 
  theFunction(f) {

}

vector<patReal> bioGaussHermite::integrate(patError*& err) {

  if (theFunction == NULL) {
    err = new patErrNullPointer("bioGhFunction") ;
    WARNING(err->describe()) ;
    return vector<patReal>() ;
  }

  vector<patReal> integral(theFunction->getSize(),0.0);
  const patReal *pA = &A[NUM_OF_POSITIVE_ZEROS];
  const patReal *px;
  
  for (px = &x[NUM_OF_POSITIVE_ZEROS - 1]; px >= x; px--) {
    vector<patReal> t1 = theFunction->getUnweightedValue(*px,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patReal>() ;
    }
    vector<patReal> t2 = theFunction->getUnweightedValue(- *px,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return vector<patReal>() ;
    }
    patReal p  = *(--pA) ;
    for (patULong i = 0 ; i < integral.size() ; ++i) {
      integral[i] +=  p * ( t1[i] + t2[i] );
    }
  }
  return integral;
}

void bioGaussHermite::Gauss_Hermite_Coefs_100pts( patReal coef[]) {
   const patReal *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
   patReal *pc = &coef[NUM_OF_ZEROS - 1];

   for (; pA >= A; pA--)  {
      *(coef++) =  *pA;
      *(pc--) = *pA;
   }   
}

void bioGaussHermite::Gauss_Hermite_Zeros_100pts( patReal zeros[] ) {
  const patReal *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
  patReal *pz = &zeros[NUM_OF_ZEROS - 1];
  
  for (; px >= x; px--)  {
    *(zeros++) = - *px;
      *(pz--) = *px;
  }   
}



