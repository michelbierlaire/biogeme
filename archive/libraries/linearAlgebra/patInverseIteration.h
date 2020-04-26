//-*-c++-*------------------------------------------------------------
//
// File name : patInverseIteration.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Aug 14 14:23:01 2003
//
//--------------------------------------------------------------------

#ifndef patInverseIteration_h
#define patInverseIteration_h

#include "patMyMatrix.h"
#include "patVariables.h"
#include "patError.h"
#include "patLu.h"

class patInverseIteration {

 public:

  /** Initialize the  matrix A.
      It is supposed that A is singular.
   */

  patInverseIteration(patMyMatrix* theMatrix) ;
  
  /**
     Perturb the diagonal of A so that A - mu I is non singular
   */
  patReal perturb(patReal mu, patError*& err) ;
  
  /**
     Performs n inverse iterations to obtain the eigenvector corresponding to the 0 eigenvalue. It will stop prematurely if A * z is close to zero
   */
  void inverseIteration(unsigned long n, patError*& err) ;

  /**
   */
  patVariables getEigenVector() ;

  /**
     Return the norm of A*z, where z is the approximation of the eigen
     vector computed by the inverse interations
   */
  patReal getAz(patError*& err) ;

 private:
  patMyMatrix* A;
  patLu lu ;
  patVariables z ;
  patReal Az ;

};

#endif


