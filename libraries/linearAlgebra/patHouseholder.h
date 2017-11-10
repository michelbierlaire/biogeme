//-*-c++-*------------------------------------------------------------
//
// File name : patHouseholder.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jun 15 16:58:39 1999
// Recoded:    Fri Jan 20 13:55:09 2006
//
//--------------------------------------------------------------------

#ifndef patHouseholder_h
#define patHouseholder_h

#include <vector>
#include "patError.h"
#include "patClass.h"
#include "patVariables.h"
#include "patMyMatrix.h"
#include "patConst.h"

/**
   Class defining Householder matrices. An Householder matrix, or Householder reflections, is defined from the vector $v$ (called the Householder vector) by
\[
P = I - \frac{2}{v^T v} v v^T
\]
The vector $Px$ is the reflection of $x$ on the hyperplane span$\{v\}^\perp$ (see Golub and Van Loan, 1996 "Matrix Computations", p 209).
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Tue Jun 15 16:58:39 1999) 
 */
class patHouseholder {

public:
  /**
   */
  patHouseholder();
  /**
   */
  patHouseholder(const patVariables& _v) ;
  /**
     Multiples the Householder matrix by a vector
  */
  patVariables operator*(const patVariables& x) const ;
  /**
     Overrides A by PA
  */
  patMyMatrix* multiply(patMyMatrix* A,
			patError*& err) const ;
  /**
   */
  patVariables* multiply(patVariables* A,
		     patError*& err) const ;
  /**
   */
  patVariables::size_type getDimension() const ;
  /**
     Given $x$, and $0 <= k <= j <= n-1$, this function computes the
   Householder matrix $P$ such that entries $k+1$ through $j$ of $Px$ are
   zero.  It is assumed that entries $k$ through $j$ of $x$ are non zero.
  */
  void setToNullify(const patVariables& x, 
		    patVariables::size_type k,
		    patVariables::size_type j,
		    patBoolean checkAssumptions,
		    patError*& err) ;
  /**
   */
  patBoolean isIdentity() ;

private:
  patVariables v ;
  patReal norm ;
  // We assume that v = [ 0 0 0 0 v_start ... v_stop 0 0 ]
  patVariables::size_type start ;
  patVariables::size_type stop ;
};

#endif
