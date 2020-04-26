//-*-c++-*------------------------------------------------------------
//
// File name : trSR1.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Apr 24 10:38:49 2001
//
//--------------------------------------------------------------------

#ifndef trSR1_h
#define trSR1_h

#include "trSecantUpdate.h"
#include "patHybridMatrix.h"
#include "patError.h"
#include "trVector.h"
#include "trParameters.h"
/**
 @doc   This class implements the quasi-Newton Symmetric rank-one update
 @author     \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Tue Apr 24 10:38:49 2001)
 */
class trSR1 : public trSecantUpdate {

public:
  /**
     Default ctor should not be used
  */
  trSR1(trParameters theParameters,
	patError*& err) ;

  /**
     Constructor: $H_0$ is the identity matrix of size "size"
  */
  trSR1(unsigned long size, 
	trParameters theParameters,
	patError*& err) ;

  /**
     Constructor: $H_0$ is a  diagonal matrix with x on the diagonal
  */
  trSR1(const trVector& x, 
	trParameters theParameters,
	patError*& err) ;
  
  /**
     Constructor: $H_0$ is given explicitly as a patHybridMatrix
   */
  trSR1(const patHybridMatrix& x,
	trParameters theParameters,
	patError*& err) ;

  /**
   */
  virtual ~trSR1() ;

  /**
     Applies the SR1 update formula.
     @param sk = currentIterate - previousIterate
     @param currentGradient
     @param previousGradient
   */
  void update(const trVector& sk,
	      const trVector& currentGradient,
	      const trVector& previousGradient,
	      ostream& str,
	      patError*& err) ;

  /**
     @return Reduced Hessian, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  virtual trSR1* getReducedHessian(vector<trBounds::patActivityStatus> status,
					    patError*& err)  ;

  /**
   */
  virtual patString getUpdateName() const  ;


  /**
   */
  virtual patReal getElement(unsigned int i, unsigned int j, patError*& err) const  ;

  virtual void print(ostream&) ;


protected:
  trSR1* submatrix ;

};


#endif








