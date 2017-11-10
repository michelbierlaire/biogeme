//-*-c++-*------------------------------------------------------------
//
// File name : patHybridMatrix
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Fri May 28 12:45:46 1999
//
//--------------------------------------------------------------------

#ifndef patHybridMatrix_h
#define patHybridMatrix_h

#include <map>
#include <list>
#include <vector>
#include <sstream>
#include <iostream>

#include "patDisplay.h"
#include "patType.h"
#include "patErrMiscError.h"
#include "patVariables.h"

class patMyMatrix ;

class patSchnabelEskow ;


/**
 @doc This class stores a lower triangular part of a matrix. This can be
 interpreted either as a symmetric matrix, an upper triangular matrix or a
 lower triangular matrix.
 @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Fri May 28 12:45:46 1999)
 */
class patHybridMatrix {

  /**
   */
  friend class patSchnabelEskow ;

  /**
   */
  friend ostream& operator<<(ostream &str, const patHybridMatrix& x) ;

public :

  /**
   */
  typedef enum  {
    patSymmetric ,
    patLower,
    patUpper,
    patDiagonal
  } patMatrixType;




  /**
     This ctor builds a diagonal matrix with vector diag on the diagonal
     @param diag vector of value to put on the diagonal
     @param err ref. of the pointer to the error object.     
  */
  patHybridMatrix(const vector<patReal>& diag,
		  patError*& err) ;  

  /**
     This constructor builds a symmetric matrix with "size" columns and rows
   */
  patHybridMatrix(vector<patReal>::size_type size) ;

  /**
     This constructor builds a symmetric matrix with "size" columns and rows, and initialize its values
   */
  patHybridMatrix(vector<patReal>::size_type size, patReal init) ;

  /**
     Copy constructor. Performs a deep copy and not a shallow copy
   */
  patHybridMatrix(const patHybridMatrix& h) ;
  

  /**
   */
  virtual ~patHybridMatrix() ;

  /**
     Initialize the matrix with a value
  */
  void init(const patReal& x) ;

  /**
     Affects a value to a cell
  */
  void setElement(vector<patReal>::size_type i, 
		  vector<patReal>::size_type j, 
		  const patReal& x,
		  patError*& err) ; 
  
  /**
     Increments the value of a cell by a given value
   */
  void addElement(vector<patReal>::size_type i, 
		  vector<patReal>::size_type j, 
		  const patReal& x,
		  patError*& err) ;


  /**
     multiply the value of a cell by a given value
   */
  void multElement(vector<patReal>::size_type i, 
		  vector<patReal>::size_type j, 
		  const patReal& x,
		  patError*& err) ;
  /**
     @param i row number
     @param j column number
     @param err ref. of the pointer to the error object.
     @return value of cell $(i,j)$
   */
  patReal getElement(vector<patReal>::size_type i, 
		     vector<patReal>::size_type j,
		     patError*& err) const ;  
  /**
     @param i row number
     @param j column number
     @return value of cell $(i,j)$
   */
  patReal operator()(vector<patReal>::size_type i,
		     vector<patReal>::size_type j,
		     patError*& err) const ;  

  /**
     Create a submatrix containing only rows and columns corresponding to
     indices in the list 
  */
  patHybridMatrix* getSubMatrix(list<vector<patReal>::size_type>,
				patError*& err) ;

  /**
     @return patTRUE if matrix is symmetric
   */
  patBoolean isSymmetric() const {
    return (type == patSymmetric || type == patDiagonal) ;
  }
  /**
     @return patTRUE if matrix is lower triangular
   */
  patBoolean isLower() const {
    return (type == patLower || type == patDiagonal) ;
  }
  /**
     @return patTRUE if matrix is upper triangular
   */
  patBoolean isUpper() const {
    return (type == patUpper || type == patDiagonal) ;
  }
  /**
     @return patTRUE if matrix is diagonal
   */
  patBoolean isDiagonal() const {

    // Note that if the type is not set to patDiagonal, the matrix may still
    // be diagonal, but the patHybridMatrix object does not check it
    // explicitly.

    return (type == patDiagonal) ;
  } 
  /**
     Imposes matrix to be interpreted as a symmetric matrix
   */
  void setSymmetric() {
    type = patSymmetric ;
  }
  /**
     Imposes matrix to be interpreted as a lower triangular matrix
  */
  void setLower() {
    type = patLower ;
  }
  /**
     Imposes matrix to be interpreted as a upper triangular matrix
   */
  void setUpper() {
    type = patUpper ;
  }
  /**
     Imposes matrix to be interpreted as a diagonal matrix
   */
  void setDiagonal() {
    type = patDiagonal ;
  }
  

  /**
     @return nuber of rows and columns of the matrix
   */
  vector<patReal>::size_type getSize() const {
    return dim ;
  } 

  /**
     Computes the Cholesky factorization of the matrix sing the Eskow
     and Schnabel technique. Warning: permutations of the rows and
     columns, as well as corrections to the diagonal may be
     performed. Check the article for details.

     @param err ref. of the pointer to the error object.  
     @return patTRUE if the factorized matrix is
     definite positive, and patFALSE if the diagonal had to be perturbed to
     obtain a definite positive matrix 
  */
  patBoolean cholesky(patReal tolSchabelEskow, patError*& err) ;

  /*
    Perform a simple cholesky factorization.
     @return patTRUE if the factorized matrix is
     definite positive, and patFALSE otherwise

   */
  patBoolean straightCholesky(patError*& err) ;

  

  /**
     Solve the system L L' x = b, where L is the lower triangular matrix
     
  */
  patVariables solve(const patVariables& b, patError*& err) ;

  /**
   */
  patBoolean isEmpty() const ;
  /**
   */
  void setType(patMatrixType t) {
    type = t ;
  }
  /**
     Defines the type as the same type as matrix x.
   */
  void setType(const patHybridMatrix& x) {
    type = x.type ;
  }
  /**
   */
  patMatrixType getType() {
    return(type) ;
  }

  /**
   */
  void dumpOnFile(const patString& fileName, 
		  patError*& err) ;

  /**
   */
  void loadFromDumpFile(const patString& fileName, 
			patError*& err) ;
  

  /**
     Implements (*this) += alpha * M
  */
  void addAlpha(patReal alpha, const patHybridMatrix& M, patError*& err) ;
  
  /**
     Implements (*this) = alpha * (*this)
  */
  void multAlpha(patReal alpha) ;
  
  /**
     Implements (*this) += alpha * M
  */
  void add(const patHybridMatrix& M, patError*& err) ;
  /**
     Implements (*this) = M
  */
  void set(const patHybridMatrix& M, patError*& err) ;


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
		     const patVariables& step,
		     patError*& err)  ;


  void resize(patULong size) ;
private:
  
  // Ddefault ctor should not be called
  patHybridMatrix() ;
  vector<patReal> data ;
  patMatrixType type ;
  vector<patReal>::size_type dim ;
public:
  vector<patVariables::size_type> pivot ;
  vector<patVariables::size_type> pivotInverse ;
private:  
  patHybridMatrix* submatrix ;
  patReal singularityPenalty ;
  patMyMatrix* Q ;
  map<patReal,patVariables> eigenVectors ;
  
private:
  vector<patReal>::size_type index(vector<patReal>::size_type i,
				   vector<patReal>::size_type j) const { 
    return ((i>j) 
	    ? j+(i*(i+1)/2) : i+(j*(j+1)/2)) ;
  }
  patVariables calcgersch(patVariables::size_type j) ;
  patBoolean final2by2(patReal tau2,
		       patReal *delta,
		       patReal gamma) ;
  
  
};

ostream& operator<<(ostream &str, const patHybridMatrix::patMatrixType& x) ;
istream& operator>>(istream &str,  patHybridMatrix::patMatrixType& x) ;

#endif
