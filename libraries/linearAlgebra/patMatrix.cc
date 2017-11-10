#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patErrMiscError.h"
#include "patMatrix.h"
#include "patVariables.h"
#include "patDisplay.h"

patMatrix transpose(const patMatrix& A, 
		    patError*&  err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patMatrix() ;
  }
  patMatrix::size_type  ma = A.size() ;
  if (ma == 0) {
    err = new patErrMiscError("A is empty") ;
    WARNING(err->describe()) ;
    return patMatrix() ;
  }
  patMatrixRow::size_type na = A[0].size() ;
  if (na == 0) {
    WARNING("A is empty") ;
    return patMatrix() ;
  }
  patMatrixRow row(ma) ;
  patMatrix res(na,row) ;
  for (patMatrix::size_type i = 0 ; i < ma ; ++i) {
    for (patMatrixRow::size_type j = 0 ; j < na ; ++j) {
      res[j][i] = A[i][j] ;
    }
  }
  return res ;
}   

patMatrix operator*(const patMatrix& A, const patMatrix& B) {
  patMatrix::size_type  ma = A.size() ;
  if (ma == 0) {
    WARNING("A is empty") ;
    return patMatrix() ;
  }
  patMatrixRow::size_type na = A[0].size() ;
  if (na == 0) {
    WARNING("A is empty") ;
    return patMatrix() ;
  }
  patMatrix::size_type  mb = B.size() ;
  if (mb == 0) {
    WARNING("B is empty") ;
    return patMatrix() ;
  }
  patMatrixRow::size_type nb = B[0].size() ;
  if (nb == 0) {
    WARNING("B is empty") ;
    return patMatrix() ;
  }
  //  DEBUG_MESSAGE(" A ("<<ma<<"x"<<na<<")*B ("<<mb<<"x"<<nb<<")") ;
  if (na != mb) {
    WARNING("Incompatible sizes in matrix multiplication") ;
    return patMatrix();
  }
  patMatrixRow resRow(nb,0.0) ;
  patMatrix    res(ma,resRow);

  for (patMatrix::size_type row = 0 ; row < ma ; ++row) {
    for (patMatrixRow::size_type column = 0 ; column < nb ; ++column) {
      for (patMatrixRow::size_type k = 0 ; k < na ; ++k) {
	res[row][column] += A[row][k] * B[k][column] ;
      }
    }
  }
  return res;
}

patVariables operator*(const patMatrix& A, const patVariables& B) {
  patMatrix::size_type  ma = A.size() ;
  if (ma == 0) {
    WARNING("A is empty") ;
    return patVariables() ;
  }
  patMatrixRow::size_type na = A[0].size() ;
  if (na == 0) {
    WARNING("A is empty") ;
    return patVariables() ;
  }
  patMatrix::size_type  mb = B.size() ;
  if (mb == 0) {
    WARNING("B is empty") ;
    return patVariables() ;
  }
  //  DEBUG_MESSAGE(" A ("<<ma<<"x"<<na<<")*B ("<<mb<<"x1)") ;
  if (na != mb) {
    WARNING("Incompatible sizes in matrix multiplication") ;
    return patVariables();
  }
  patVariables res(ma,0.0);

  for (patMatrix::size_type row = 0 ; row < ma ; ++row) {
    for (patMatrixRow::size_type k = 0 ; k < na ; ++k) {
      res[row] += A[row][k] * B[k] ;
    }
  }
  return res;
}

ostream& operator<<(ostream &str, const patMatrix& x) {
  if (x.size() == 0) return(str) ;
  str << "[" ;
  for (patMatrix::const_iterator i = x.begin() ;
       i != x.end() ;
       ++i) {
    str << *i << endl ;
  }
  str << "]" ;
  return(str); 
}
