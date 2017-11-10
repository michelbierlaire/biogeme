//-*-c++-*------------------------------------------------------------
//
// File name : trHybridMatrix.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Jun  8 16:55:07 2000
//
//--------------------------------------------------------------------

#ifndef trHybridMatrix_h
#define trHybridMatrix_h

#include "patHybridMatrix.h"
#include "trMatrixVector.h"

/**
 @doc This class encapsulates patHybridMatrix to comply with the 
 trMatrixVector interface.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Thu Jun  8 16:55:07 2000)
 */
class trHybridMatrix : public trMatrixVector {

public :  
  /**
   */
  trHybridMatrix(patHybridMatrix* _mPtr) ;

  /**
   */
  ~trHybridMatrix() ;
  /**
   */
  trVector operator()(const trVector& x, 
		      patError*& err) ;
  /**
   */
  patBoolean providesPreconditionner() const ;
  
  /**
    This function provides a preconditionner and allocates memory. The caller
    is responsible for releasing the memory. 
  */
  trPrecond* 
  createPreconditionner(patError*& err) const ;


  /**
     @return Corresponding object for the reduced matrix, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  virtual trMatrixVector* 
  getReduced(vector<trBounds::patActivityStatus> status,
	     patError*& err)   ;

  
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

  /**
   */
  virtual void print(ostream&) ;
private :

  patHybridMatrix* hMatrix ;
  trHybridMatrix* submatrix ;
};

#endif
