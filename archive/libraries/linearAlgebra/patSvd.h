//-*-c++-*------------------------------------------------------------
//
// File name : patSvd.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Aug 15 11:21:09 2005
//
//--------------------------------------------------------------------

#ifndef patSvd_h
#define patSvd_h

#include <map>
#include "patError.h"
#include "patVariables.h"

class patMyMatrix ;

/**
   Compute the SVD decomposition of a matrix A = U W V', where U and V
   are orthogonal and W is diagonal. A is replaced by U.
 */

class patSvd {

 public:
  patSvd(patMyMatrix* aMat) ;
  ~patSvd() ;
  void computeSvd(patULong maxIter, 
		  patError*& err) ;
  patVariables backSubstitution(const patVariables& b) ;
  const patMyMatrix* getU() const ;
  const patMyMatrix* getV() const ;
  const patVariables* getSingularValues() const ;
  const patMyMatrix* computeInverse() ;
  const patMyMatrix* getInverse() const ;
  map<patReal,patVariables>  getEigenVectorsOfZeroEigenValues(patReal threshold) ; // patParameters::the()->getgevSingularValueThreshold()
  patReal getSmallestSingularValue() ;
 private:
  patMyMatrix* theMatrix ;
  patMyMatrix* V ;
  patVariables* W ;
  patMyMatrix* theInverse ;
  patVariables* rv1 ;
  patBoolean svdComputed ;
  unsigned long m ;
  unsigned long n ;
  patReal smallestSingularValue ;

};

#endif
