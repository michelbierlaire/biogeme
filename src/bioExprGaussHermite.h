//-*-c++-*------------------------------------------------------------
//
// File name : bioExprGaussHermite.h
// Author :    Michel Bierlaire
// Date :      Sat May 21 19:22:23 2011
// Modified for biogemepython 3.0: Wed May  9 17:51:22 2018
//
//--------------------------------------------------------------------

#ifndef bioExprGaussHermite_h
#define bioExprGaussHermite_h

#include "bioGaussHermite.h"

class bioExpression ;

// This object is designed to gather all variables that must be
// integrated into one vector.  It is assumed that, if the hessian is
// requested, so is the gradient.

class bioExprGaussHermite: public bioGhFunction {
 public:
  bioExprGaussHermite(bioExpression* e, std::vector<bioUInt> derivl, bioUInt l, bioBoolean wg, bioBoolean wh) ;
  std::vector<bioReal> getValue(bioReal x) ;
  bioUInt getSize() const ;
private:
  bioBoolean withGradient ;
  bioBoolean withHessian ;
  bioExpression* theExpression ;
  std::vector<bioUInt> derivLiteralIds ;
  bioUInt rvId;
  bioReal theValue ;
};

#endif
