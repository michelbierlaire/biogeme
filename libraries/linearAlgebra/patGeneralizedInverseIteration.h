//-*-c++-*------------------------------------------------------------
//
// File name : patGeneralizedInverseIteration.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jan 20 12:54:32 2006
//
//--------------------------------------------------------------------

#ifndef patGeneralizedInverseIteration_h
#define patGeneralizedInverseIteration_h

#include "patType.h"
#include "patError.h"
#include "patMyMatrix.h"

class patLu ;

class patGeneralizedInverseIteration {

 public:

  patGeneralizedInverseIteration(const patMyMatrix* aMatrix, 
				 patMyMatrix* initialEigenVectors,
				 patError*& err) ;
  
  ~patGeneralizedInverseIteration() ;

  patMyMatrix* computeEigenVectors(patULong maxIter, // patParameters::the()->getgevInverseIteration()
				   patError*& err) ;
  
 protected:
  patReal performOneIteration(patError*& err) ;
  void computeShiftedLu(patError*& err) ;

 private:
  const patMyMatrix* theMatrix ;
  patMyMatrix* theResult ;
  patMyMatrix Lu ;
  patLu* theLuDecomposition ;
};


#endif
