//-*-c++-*------------------------------------------------------------
//
// File name : trTheoryGCP.h
// Date :      Mon Nov 27 09:21:06 2000
//
//--------------------------------------------------------------------

#ifndef trTheoryGCP_h
#define trTheoryGCP_h

#include "trBounds.h"
#include "trMatrixVector.h"
#include "trParameters.h"

/**
 @doc  This class encapsulates Algorithm 12.2.2 of Conn, Gould, Toint (2000) "Search for the Generalized Cauchy Point", for the quadratic model 
   \[
   g^T s + \frac{1}{2} s^T H s
   \].
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Nov 27 09:21:06 2000) 
 @see Conn, Gould Toint (2000) Trust Region Methods, SIAM
 */
class trTheoryGCP {

public:
  /**
     Constructor
     @param _radius radius of the trust region
     @param _currentIterate current iterate $x_k$
     @param _g pointer to the vector $g$ defining the linear term of the 
               quadratic model
     @param _H pointer to the matrix $H$ defining the quadratic term of the 
               quadratic model
  */
  trTheoryGCP(const trBounds& _bounds,
	      patReal _radius,
	      const trVector& _currentIterate,
	      const trVector& _g,
	      trMatrixVector& _H,
	      trParameters theParameters) ;
		   
  /**
   */
  trVector computeGCP(patError*& err) ;

private:
  /**
     Check that conditions (12.2.1) are verified. 
   */
  patBoolean checkConditions1221(patError*& err) ;
  /**
     Check that at least one of (12.2.2) is verified. 
   */
  patBoolean checkConditions1222(patError*& err) ;
  /**
   */
  void computeProjections(patError*& err) ;
private:
  const trBounds& bounds ;
  patReal radius ;
  const trVector& currentIterate ;
  const trVector& g ;
  trMatrixVector& H ;

  patReal tmax ;
  patReal tmin ;
  patReal t ;

  trVector projection ;
  trVector sk ;

  patReal normSk ;
  patReal gksk ;

  // Here, mk is defined as m(projection)-m(currentIterate) = s_k^T g_k +
  // s_k^T H s_k. Note that, in the book, mk is the value of the quadratic
  // model m(projection).
  patReal mk ;

  trParameters theParameters ;
};  
#endif
