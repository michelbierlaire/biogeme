//-*-c++-*------------------------------------------------------------
//
// File name : bioGhFunction.h
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 11:54:05 2010
//
//--------------------------------------------------------------------

#ifndef bioGhFunction_h
#define bioGhFunction_h

#include "patError.h"
#include "patConst.h"
#include "patType.h" 

class bioGhFunction {
  friend class bioGaussHermite ;
 public:
  bioGhFunction() ;
  virtual patULong getSize() const = PURE_VIRTUAL ;
  virtual vector<patReal> getValue(patReal x, patError*& err) = PURE_VIRTUAL ;
 protected:
  // This function multiplies the function to be integrated by exp(x*x). 
  virtual vector<patReal> getUnweightedValue(patReal x, patError*& err) ;

};

#endif
