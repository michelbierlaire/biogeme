//-*-c++-*------------------------------------------------------------
//
// File name : trTointSteihaug.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 19 14:30:59 2000
//
// Implementation of algorithme 7.5.1: the Steihaug-Toint truncated CG 
// method
//
// Source: Conn, Gould Toint (2000) Trust Region Methods
//--------------------------------------------------------------------

#ifndef trTointSteihaug_h
#define trTointSteihaug_h

#include "patError.h"
#include "patType.h"
#include "trVector.h" 
#include "trParameters.h"

class trMatrixVector ;
class trPrecond ;

/**
 @doc Implementation of algorithme 7.5.1: the Steihaug-Toint truncated CG 
 method to solve the trust-region problem 
 \[
 \min_s g^{T} s  + \frac{1}{2} s^{T} Hs
 \]
 subject to $\|s\|_M \leq r$, where $r$ is the radius of the trust region, and $\|s\|_M = s^T M s$ is a norm defined by the preconditioner $M$, a symmetric positive definite matrix.
 @author     \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Wed Jan 19 14:30:59 2000)
 @see Conn, Gould Toint (2000) Trust Region Methods, SIAM
*/
class trTointSteihaug {
  
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
     @param _g pointer to the $g$ vector of the quadratic model
     @param _H pointer to the $H$ matrix of the quadratic model
     @param _radius radius of the trust region
     @param _m pointer to the preconditioner
     @param err ref. of the pointer to the error object.
   */
  trTointSteihaug(const  trVector* _g,
		  trMatrixVector* _H,
		  patReal _radius,
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
     Runs the algortihm
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
     @return norm of the CG step 
   */
  patReal getNormStep() ;
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
  const trVector* g ;
  patReal normMsk2 ;
  trMatrixVector* H ;
  const trPrecond* M ;
  patReal radius ;
  trTermStatus status ;
  trVector solution ;
  trParameters theParameters ;
  vector<patString> statusName ;

  patReal normG ;

  patReal kfgr ;
  patReal theta ;

  int maxIter ;
  

};
#endif
