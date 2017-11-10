//-*-c++-*------------------------------------------------------------
//
// File name : trPrecond.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 19 16:07:56 2000
//
//--------------------------------------------------------------------

#ifndef trPrecond_h
#define trPrecond_h

#include "patConst.h"
#include "trVector.h"
#include "patError.h"

#include "patHybridMatrix.h"

/**
   @doc This class provides the identity as
   a preconditioner. Its main purpose is to define an appropriate interface to
   be used by the algorithm. An actual preconditioner must derive from this
   class. 
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jan 19 16:07:56 2000)
*/


class trPrecond {
public:
  /**
   */
  virtual ~trPrecond() {} ;
  /**
     If A is the preconditionner, this function returns $x$ such that $Ax=b$.
     If this parent class is used, $b$ is returned as is.
  */
  virtual trVector solve(const trVector* b,
			 patError*& err) const  ;


  virtual patHybridMatrix getPrecond() { return patHybridMatrix(0); } ;

};

#endif


