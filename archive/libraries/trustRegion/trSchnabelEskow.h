//-*-c++-*------------------------------------------------------------
//
// File name : trSchnabelEskow.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri May 28 11:15:22 1999
//
//--------------------------------------------------------------------

#ifndef trSchnabelEskow_h
#define trSchnabelEskow_h

#include "patError.h"
#include "trPrecond.h"
#include "patHybridMatrix.h"


/**
  @doc Implements a preconditioner based on a (possibly incomplete) Cholesky factorization, as described by Schnabel & Eskow (1991) 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Fri May 28 11:15:22 1999)
  @see Schnabel & Eskow (1991) A new modified Cholesky factorization \emph{SIAM Journal on Scientific and Statistical Computing} 11, pp. 1136--1158.
 */
class trSchnabelEskow: public trPrecond {

private:

  patHybridMatrix precond ;


public:
  /**
   */
  friend ostream& operator<<(ostream &str, const trSchnabelEskow& x) ;
  /**
     @param p symmetric matrix to be factorized
   */
  trSchnabelEskow(const patHybridMatrix& p) ;
  /**
     @param err ref. of the pointer to the error object.
   */
  void factorize(patReal tolerance, // patParameters::getTolSchnabelEskow()
		 patError*& err);
  /**
     This must be called after the factorization has been completed. It solves
     a linear system of equations $Ax = b$ where we know $L$ such that $P^TAP
     = LL^T$, where $P$ is a permutation matrix, and $L$ a lower triangular
     matrix (the Cholesky factor of $A$)
     @see "Matrix Computations" by G.H. Golub and C. F. Van Loan, North
     Oxford Academic, 1983 
     @param b As input, points to $b$
     @return solution of $Ax=b$
  */
  virtual trVector solve(const trVector* b,
			 patError*& err) const ;


  virtual patHybridMatrix getPrecond() { return precond ; } ;
  
};




#endif
