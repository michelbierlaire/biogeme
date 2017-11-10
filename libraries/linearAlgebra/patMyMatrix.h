//-*-c++-*------------------------------------------------------------
//
// File name : patMyMatrix.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:33:12 2005
//
//--------------------------------------------------------------------

#ifndef patMyMatrix_h
#define patMyMatrix_h

#include "patType.h"
#include "patVariables.h"
#include "patDisplay.h"

#include "patHybridMatrix.h" 
/**
   Implementation of a matrix
 */

class patMyMatrix {

  friend ostream& operator<<(ostream &str, const patMyMatrix& x) ;
  friend void mult(const patMyMatrix& a, const patMyMatrix& b, patMyMatrix& res, patError*& err) ;
  friend void multABTransp(const patMyMatrix& a, const patMyMatrix& btransp, patMyMatrix& res, patError*& err) ;
  friend void multATranspB(const patMyMatrix& a, const patMyMatrix& btransp, patMyMatrix& res, patError*& err) ;
  friend void multVec(const patMyMatrix& a, const patVariables& b, patVariables& res, patError*& err) ;
  friend void multTranspVec(const patMyMatrix& a, const patVariables& b, patVariables& res, patError*& err) ;
 public:
  patMyMatrix() ;
  patMyMatrix(const patHybridMatrix& aHybridMatrix, patError*& err) ;
  patMyMatrix(const patMyMatrix& aMat) ;
  patMyMatrix(unsigned long n, unsigned long m) ;
  patMyMatrix(unsigned long n, unsigned long m, patReal init) ;
  ~patMyMatrix() ;

  patMyMatrix& operator=(const patMyMatrix& a) ;

  patVariables getRow(unsigned long r) ;
  patVariables getColumn(unsigned long r) ;

  inline patReal* operator[](const unsigned long i) { 
#ifdef patDEBUG_ON
    if (i >= nr) {
      WARNING("Out of bounds: " << i << ">=" << nr) ; 
    }
#endif
    return data[i] ; 
  }
  inline const patReal* operator[](const unsigned long i) const {
#ifdef patDEBUG_ON
    if (i >= nr) {
      WARNING("Out of bounds: " << i << ">=" << nr) ; 
    }  
#endif
    return data[i]; 
  }
  inline unsigned long nRows() const { return nr ; }
  inline unsigned long nCols() const { return nc ; }

 private:
  unsigned long nr ;
  unsigned long nc ;
  patReal **data ;
  
};

#endif
