//-*-c++-*------------------------------------------------------------
//
// File name : trMatrixVector.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Jan 19 16:06:10 2000
//
//--------------------------------------------------------------------

#ifndef trMatrixVector_h
#define trMatrixVector_h

#include <iostream>
#include "patConst.h"
#include "trVector.h"
#include "patError.h"
#include "trBounds.h"

class trPrecond ;


/**
  @doc This class implements an interface for matrix object that are not stored
   explicitly, but are able to provide a matrix-vector product. This is all
   that is required for a CG-like algorithm.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed Jan 19 16:06:10 2000)
  */
class trMatrixVector {
public:
  /**
   */
  virtual ~trMatrixVector() {}
  /**
   */
  virtual trVector operator()(const trVector& x, 
			      patError*& err)  = PURE_VIRTUAL;
  /**
     If the matrix can be used to provide a preconditionner, this method shoud return patTRUE
   */
  virtual patBoolean providesPreconditionner() const = PURE_VIRTUAL ;

  /**
     @return Corresponding object for the reduced matrix, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  virtual trMatrixVector* 
  getReduced(vector<trBounds::patActivityStatus> status,
	     patError*& err) 
    = PURE_VIRTUAL ;

  /**
     @return patTRUE is correction has been done successfully. patFALSE otherwise
     Implements Michael's algorithm for penalties assocaited with singular subspce
   */
  virtual patBoolean correctForSingularity(int svdMaxIter, // patParameters::the()->getsvdMaxIter()
					   patReal threshold, // patParameters::the()->getgevSingularValueThreshold()
					   patError*& err) = PURE_VIRTUAL ;
  
  /**
     Implements Michael's algorithm for penalties assocaited with singular subspce
   */
  virtual void updatePenalty(patReal singularityThreshold, // patParameters::the()->BTRSingularityThreshold()
			     const trVector& step,patError*& err) = PURE_VIRTUAL ;

  /**
     This function provide the preconditionner (typically, a Cholesky factor
     of a symmetric def pos matrix). 
   */
  virtual trPrecond* 
  createPreconditionner(patError*& err) const 
    = PURE_VIRTUAL;
  
  virtual void print(ostream&) = PURE_VIRTUAL;

};


#endif


