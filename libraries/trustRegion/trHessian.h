//-*-c++-*------------------------------------------------------------
//
// File name : trHessian.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri Jan 21 15:13:26 2000
//
//--------------------------------------------------------------------

#ifndef trHessian_h
#define trHessian_h

#include "trMatrixVector.h"
#include "patHybridMatrix.h"
#include "trBounds.h"
#include "patMyMatrix.h"
#include "trParameters.h"

/**
  @doc  This class implements a trMatrixVector based on the hessian of a non linear
   function, or a BFGS approximation of it.
   @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri Jan 21 15:13:26 2000)
*/
class trHessian : public trMatrixVector {

public:

  /**
   */
  friend class trBFGS ;

  /**
     @param size nuber of rows and columns
   */
  trHessian(trParameters theParameters,
	    unsigned long size) ;
  /**
     Copy constructor
   */
  trHessian(const trHessian& h) ;
  /**
     Dtor
   */
  virtual ~trHessian() ;

  /**
   */
  void copy(const trHessian& h) ;
  
  /**
     Computes the matrix-vector product
   */
  virtual trVector operator()(const trVector& x, 
			      patError*& err)  ;
  /**
     @return patTRUE
   */
  virtual patBoolean providesPreconditionner() const ;

  /**
   This function allocates memory. The caller is responsible for releasing
   the memory.
   */
  virtual trPrecond* 
  createPreconditionner(patError*& err) const ;

  /**
     Affects value $x$ to cell $(i,j)$.
   */
  void setElement(unsigned long i, 
		  unsigned long j, 
		  patReal x,
		  patError*& err) ;

  /**
     Add value $x$ to cell $(i,j)$.
   */
  void addElement(unsigned long i, 
		  unsigned long j, 
		  patReal x,
		  patError*& err) ;

  /**
     Multiply  cell $(i,j)$ by value $x$.
   */
  void multElement(unsigned long i, 
		  unsigned long j, 
		  patReal x,
		  patError*& err) ;

  /**
     Multiply  all entries by value $x$.
   */
  void multAllEntries(patReal x,
		      patError*& err) ;


  patReal getElement(unsigned int i, 
		  unsigned int j, 
		  patError*& err) ;



  /**
   */
  unsigned long getDimension() const ;

  /**
   */
  friend  ostream& operator<<(ostream &str, const trHessian& x) ;

  /**
    @return pointer to the original patHybridMatrix object
   */
  patHybridMatrix* getHybridMatrixPtr() {
    return &matrix ;
  }

    /**
       @return Object for MTL
     */

    patMyMatrix getMatrixForLinAlgPackage(patError*& err) const ;
  //    LaGenMatDouble getMatrixForLinAlgPackage(patError*& err) const ;

  

  /**
     @return Reduced Hessian, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
   */
  trHessian* getReducedHessian(vector<trBounds::patActivityStatus> status,
			      patError*& err) ;


  /**
     Important: the caller is responsible for releasing the memory allocated by this function.
     @return Corresponding object for the reduced matrix, that is the submatrix corresponding to free variables
     @param status vector describing the status of the variables. Only variables with status patFree will be considered.
  */
  trMatrixVector* getReduced(vector<trBounds::patActivityStatus> status,
			     patError*& err)  ;

  /**
     Implements (*this) += alpha * M
   */
  void add(patReal alpha, const trHessian& M, patError*& err) ;

  /**
     Implements (*this) =  M
   */
  void set(const trHessian& M, patError*& err) ;


  /**
     Set all entries to 0
   */
  void setToZero() ;

  /**
     Set all entries to 0, except on the diagonal that contains 1.0
   */
  void setToIdentity(patError*& err) ;


  /**
   */
  void changeSign() ;
  
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

  void resize(patULong size) ;
private:

  trParameters theParameters ;
  patHybridMatrix matrix ;
  trHessian* submatrix ;


};

#endif
