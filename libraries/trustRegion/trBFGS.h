//-*-c++-*------------------------------------------------------------
//
// File name : trBFGS.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 21 15:17:39 2000
//
//--------------------------------------------------------------------

#ifndef trBFGS_h
#define trBFGS_h

#include "trSecantUpdate.h"
#include "patHybridMatrix.h"
#include "patError.h"
#include "trVector.h"
#include "trParameters.h"

/**
  @doc  This class implements the quasi-Newton BFGS update
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Jan 21 15:17:39 2000)
 */
class trBFGS : public trSecantUpdate {

public:
  /**
     Constructor: $H_0$ is the identity matrix of size "size"
  */
  trBFGS(unsigned long size, 
	 trParameters theParameters,
	 patError*& err) ;

  /**
     Constructor: $H_0$ is a  diagonal matrix with x on the diagonal
  */
  trBFGS(const trVector& x, 
	 trParameters theParameters,
	 patError*& err) ;
  
  /**
     Constructor: $H_0$ is given explicitly as a patHybridMatrix
   */
  trBFGS(const patHybridMatrix& x, 
	 trParameters theParameters,
	 patError*& err) ;

  /**
   */
  virtual ~trBFGS() ;

  /**
     Applies the BFGS update formula.
     @param sk = currentIterate - previousIterate
     @param currentGradient
     @param previousGradient
   */
  virtual void update(const trVector& sk,
		      const trVector& currentGradient,
		      const trVector& previousGradient,
		      ostream& str,
		      patError*& err) ;

  /**
     @return Reduced Hessian, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  virtual trBFGS* getReducedHessian(vector<trBounds::patActivityStatus> status,
				    patError*& err)  ;

  /**
   */
  virtual patString getUpdateName() const  ;



  /**
   */
  virtual patReal getElement(unsigned int i, unsigned int j, patError*& err) const  ;

  virtual void print(ostream&) ;


protected:
  trBFGS* submatrix ;

};


#endif








