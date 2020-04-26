//-*-c++-*------------------------------------------------------------
//
// File name : bioGhFunction.h
// Author :    Michel Bierlaire
// Date :      Thu Apr  8 11:54:05 2010
// Modified for biogemepython 3.0: Wed May  9 16:12:27 2018
//
//--------------------------------------------------------------------

#ifndef bioGhFunction_h
#define bioGhFunction_h

#include <vector>
#include "bioTypes.h"
#include "bioConst.h"

class bioGhFunction {
  friend class bioGaussHermite ;
 public:
  bioGhFunction() ;
  virtual bioUInt getSize() const = PURE_VIRTUAL ;
  virtual std::vector<bioReal> getValue(bioReal x) = PURE_VIRTUAL ;
 protected:
  // This function multiplies the function to be integrated by exp(x*x). 
  virtual std::vector<bioReal> getUnweightedValue(bioReal x) ;

};

#endif
