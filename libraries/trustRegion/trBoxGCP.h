//-*-c++-*------------------------------------------------------------
//
// File name : trBoxGCP.h
// Date :      Mon Nov 27 12:44:24 2000
//
//--------------------------------------------------------------------

#ifndef trBoxGCP_h
#define trBoxGCP_h

#include "trBounds.h"
#include "trMatrixVector.h"

/**
  @doc This class encapsulates algorithm 17.3.1 (p. 791) from Conn, Gould and Toint (2000). It computes the generalized Cauchy point for simple bounds constraints and a box-shaped trust region (based on $\|\cdot\|_{\infty}$).
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Nov 27 12:44:24 2000)
 @see Conn, Gould Toint (2000) Trust Region Methods, SIAM

 */

class trBoxGCP {

  public:  
  /**
     Constructor
     @param _radius radius of the trust region
     @param _currentIterate current iterate $x_k$
     @param _direction direction in which the GCP is computed. Usually -g
     @param _g pointer to the vector $g$ defining the linear term of the 
               quadratic model
     @param _H pointer to the matrix $H$ defining the quadratic term of the 
               quadratic model
  */
  trBoxGCP(const trBounds& _bounds,
	   patReal _radius,
	   const trVector& _currentIterate,
	   const trVector& _direction,
	   const trVector& _g,
	   trMatrixVector& _H) ;
  /**
   */
  trVector computeGCP(patULong maxGcpIter, // patParameters::the()->getBTRMaxGcpIter() ;
		      patError*& err) ;

private:
  const trBounds& bounds ;
  patReal radius ;
  const trVector& currentIterate ;
  const trVector& direction ;
  const trVector& g ;
  trMatrixVector& H ;

};

#endif
