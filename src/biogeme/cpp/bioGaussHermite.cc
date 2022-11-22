//-*-c++-*------------------------------------------------------------
//
// File name : bioGaussHermite.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 10:57:03 2010
// Modified for biogemepython 3.0: Wed May  9 16:21:08 2018
//
//-------------------------------------------------------------------

#include "bioGaussHermite.h"
#include "bioExceptions.h"

bioGaussHermite::bioGaussHermite(bioGhFunction* f) : 
  theFunction(f) {

}

std::vector<bioReal> bioGaussHermite::integrate() {

  if (theFunction == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"Function to integrate.") ;
  }

  std::vector<bioReal> integral(theFunction->getSize(),0.0);
  const bioReal *pA = &A[NUM_OF_POSITIVE_ZEROS];
  const bioReal *px;
  
  for (px = &x[NUM_OF_POSITIVE_ZEROS - 1]; px >= x; px--) {
    std::vector<bioReal> t1 = theFunction->getUnweightedValue(*px) ;
    std::vector<bioReal> t2 = theFunction->getUnweightedValue(- *px) ;
    bioReal p  = *(--pA) ;
    for (bioUInt i = 0 ; i < integral.size() ; ++i) {
      integral[i] +=  p * ( t1[i] + t2[i] );
    }
  }
  return integral;
}

void bioGaussHermite::Gauss_Hermite_Coefs_100pts( bioReal coef[]) {
   const bioReal *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
   bioReal *pc = &coef[NUM_OF_ZEROS - 1];

   for (; pA >= A; pA--)  {
      *(coef++) =  *pA;
      *(pc--) = *pA;
   }   
}

void bioGaussHermite::Gauss_Hermite_Zeros_100pts( bioReal zeros[] ) {
  const bioReal *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
  bioReal *pz = &zeros[NUM_OF_ZEROS - 1];
  
  for (; px >= x; px--)  {
    *(zeros++) = - *px;
      *(pz--) = *px;
  }   
}



