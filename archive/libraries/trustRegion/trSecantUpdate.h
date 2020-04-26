//-*-c++-*------------------------------------------------------------
//
// File name : trSecantUpdate.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 21 15:17:39 2000
//
//--------------------------------------------------------------------

#ifndef trSecantUpdate_h
#define trSecantUpdate_h

#include "trMatrixVector.h"
#include "patHybridMatrix.h"
#include "patError.h"
#include "trVector.h"
#include "trParameters.h"

/**
 @doc   This class defines a generic secant update. The actual update computation is purely virtual
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi} (Fri Jan 21 15:17:39 2000)
 */
class trSecantUpdate : public trMatrixVector {

  /**
   */
  friend ostream& operator<<(ostream &str, const trSecantUpdate& x) { str << x.matrix ; return str ;}
public:

  /**
     Constructor: $H_0$ is the identity matrix of size "size"
  */
  trSecantUpdate(unsigned long size, 
		 trParameters theParameters,
		 patError*& err) ;

  /**
     Constructor: $H_0$ is a  diagonal matrix with x on the diagonal
  */
  trSecantUpdate(const trVector& x, 
		 trParameters theParameters,
		 patError*& err) ;
  
  /**
     Constructor: $H_0$ is given explicitly as a patHybridMatrix
   */
  trSecantUpdate(const patHybridMatrix& x, 
		 trParameters theParameters,
		 patError*& err) ;

  /**
   */
  virtual ~trSecantUpdate() ;

  /**
     Computes $Hx$, where $H$ is the quasi-newton matrix
   */
  virtual trVector operator()(const trVector& x, 
			      patError*& err)  ;
  /**
     @return patTRUE
   */
  virtual patBoolean providesPreconditionner() const ;

  /**
     Provides the Schnabel Eskow preconditioner. This function allocates
     memory. The caller is responsible for releasing it 
  */

  virtual trPrecond* 
  createPreconditionner(patError*& err) const ;

  /**
     Applies the  update formula.
     @param sk = currentIterate - previousIterate
     @param currentGradient
     @param previousGradient
   */
  virtual void update(const trVector& sk,
		      const trVector& currentGradient,
		      const trVector& previousGradient,
		      ostream& str,
		      patError*& err) = PURE_VIRTUAL;

  /**
     @return Reduced Hessian, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  virtual trSecantUpdate* getReducedHessian(vector<trBounds::patActivityStatus> status,
					    patError*& err) = PURE_VIRTUAL ;


  /**
     Important: the caller is responsible for releasing the memory allocated by this function.
     @return Corresponding object for the reduced matrix, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
  */
  virtual trMatrixVector* getReduced(vector<trBounds::patActivityStatus> status,
			     patError*& err)  ;

  /**
   */
  virtual patString getUpdateName() const = PURE_VIRTUAL ;

  /**
     @return patTRUE is correction has been done successfully. patFALSE otherwise
     Implements Michael's algorithm for penalties assocaited with singular subspce
  */
  patBoolean correctForSingularity(int svdMaxIter, // patParameters::the()->getsvdMaxIter()
				   patReal threshold, // patParameters::the()->getgevSingularValueThreshold()
				   patError*& err) ;
  
  /**
     Implements Michael's algorithm for penalties assocaited with singular subspce
   */
  void updatePenalty(patReal singularityThreshold, // patParameters::the()->BTRSingularityThreshold()
		     const trVector& step,
		     patError*& err)  ;

protected:
  
  patHybridMatrix matrix ;
  trParameters theParameters ;

private:
  /**
     Default ctor should not be used
  */
  trSecantUpdate() ;

};


#endif








