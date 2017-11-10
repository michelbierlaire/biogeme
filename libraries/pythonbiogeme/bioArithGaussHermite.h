//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGaussHermite.h
// Author :    Michel Bierlaire
// Date :      Sat May 21 19:22:23 2011
//
//--------------------------------------------------------------------

#ifndef bioArithGaussHermite_h
#define bioArithGaussHermite_h

#include "bioGaussHermite.h"

class bioExpression ;

// It is assumed that, if the hessian is requested, so is the gradient.

class bioArithGaussHermite: public bioGhFunction {
 public:
  bioArithGaussHermite(bioExpression* e, vector<patULong> derivl, patULong l, patBoolean wg, patBoolean wh) ;
  vector<patReal> getValue(patReal x, patError*& err) ;
  patULong getSize() const ;
  patULong getThreadId() const ;
private:
  patBoolean withGradient ;
  patBoolean withHessian ;
  bioExpression* theExpression ;
  vector<patULong> derivLiteralIds ;
  patULong literalId; 
};

#endif
