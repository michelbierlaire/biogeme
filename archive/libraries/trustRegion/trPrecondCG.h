//-*-c++-*------------------------------------------------------------
//
// File name : trPrecondCG.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Nov 27 16:53:49 2000
//
// Implementation of algorithme 5.1.4: the preconditioned CG 
// method
//
// Source: Conn, Gould Toint (2000) Trust Region Methods
//--------------------------------------------------------------------

#ifndef trPrecondCG_h
#define trPrecondCG_h

#include "patError.h"
#include "patType.h"
#include "trVector.h" 
#include "trBounds.h"
#include "trParameters.h"

class trMatrixVector ;
class trPrecond ;

/**
@doc Implementation of algorithme 5.1.4: the preconditioned conjugate gradient method to solve the quadratic minimization problem
 \[
 \min_s g^{T} s  + \frac{1}{2} s^{T} Hs
 \]
 The iterations are stopped as soon as an iterate violate the bounds constraints.
 @author     \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Mon Nov 27 16:53:49 2000)
 @see Conn, Gould Toint (2000) Trust Region Methods, SIAM, p. 88
*/
class trPrecondCG {
  
public :
  /**
   */
   enum trTermStatus {
     /**
      */
     trUNKNOWN = 0, 
     /**
      */
     trNEG_CURV = 1, 
     /**
      */
     trOUT_OF_TR = 2, 
     /**
      */
     trMAXITER = 3, 
     /**
      */
     trCONV = 4}   ;
  
  /**
     @param _g $g$ vector of the quadratic model
     @param _H pointer to the $H$ matrix of the quadratic model
     @param _bounds definition of the feasible domain for the step
     @param _m pointer to the preconditioner
     @param err ref. of the pointer to the error object.
   */
  trPrecondCG(const trVector& _g,
	      trMatrixVector* _H,
	      const trBounds& _bounds,
	      const trPrecond* _m,
	      trParameters theParameters,
	      patError*& err) ;
				   
  /**
     @param err ref. of the pointer to the error object.
     @return termination status
     \begin{description}
     \item[trUNKNOWN] undefined, yet
     \item[trNEG_CURV] a direction of negative curvature has been detected
     \item[trOUT_OF_TR] the current iterate is out of the trust region
     \item[trMAXITER] the maximum number of iterations has been reached
     \item[trCONV] convergence has been reached
     \end{description}
   */
  
  trTermStatus getTermStatus(patError*& err) ;
  
  /**
     Runs the algorithm
     @param err ref. of the pointer to the error object.
   */
  trTermStatus run(patError*& err) ;

  /**
     @param err ref. of the pointer to the error object. 
     @return current
     iterate. Usually called when the iterations are finished to get the
     obtained solution
  */
  trVector getSolution(patError*& err) ;

  /**
     @param err ref. of the pointer to the error object. 
     @return value of the model $s^T g + \frac{1}{2}s^THs$ at the current iterate $s$
  */
  patReal getValue(patError*& err) ;

  /**
     @return 8-character string describing the termination status
     \begin{description}
     \item[trUNKNOWN] returns "Unknown "
     \item[trNEG_CURV] returns "NegCurv " 
     \item[trOUT_OF_TR] returns "OutTrReg" 
     \item[trMAXITER] returns "MaxIter " 
     \item[trCONV] returns "Converg " 
     \end{description}
   */
  patString getStatusName(trTermStatus status) ;

private :
  const trVector& g ;
  trMatrixVector* H ;
  const trPrecond* M ;
  const trBounds& bounds ;
  trTermStatus status ;
  trParameters theParameters ;
  trVector solution ;
  vector<patString> statusName ;


  patReal normG ;

  patULong maxIter ;
};
#endif
