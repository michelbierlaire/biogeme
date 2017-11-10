//-*-c++-*------------------------------------------------------------
//
// File name : patMyMatrix.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Jun 16 09:40:20 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <iostream>

#include "patMath.h"

#include "patMyMatrix.h"
#include "patDisplay.h"

patMyMatrix::patMyMatrix() : nr(0), nc(0), data(NULL) {

}

patMyMatrix::patMyMatrix(const patHybridMatrix& aMat, patError*& err) : 
  nr(aMat.getSize()),
  nc(aMat.getSize()),
  data(new patReal*[nr])
{
  unsigned long i ;
  unsigned long j ;
  data[0] = new patReal[nr * nc] ;
  for (i = 1 ; i < nr ; ++i) {
    data[i] = data[i-1] + nc ;
  }
  for (i = 0 ; i < nr ; ++i) {
    for (j = 0 ; j < nc ; ++j) {
      data[i][j] = aMat.getElement(i,j,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
}


patMyMatrix::patMyMatrix(const patMyMatrix& aMat) : 
  nr(aMat.nr),
  nc(aMat.nc),
  data(new patReal*[nr])
{
  unsigned long i ;
  unsigned long j ;
  data[0] = new patReal[nr * nc] ;
  for (i = 1 ; i < nr ; ++i) {
    data[i] = data[i-1] + nc ;
  }
  for (i = 0 ; i < nr ; ++i) {
    for (j = 0 ; j < nc ; ++j) {
      data[i][j] = aMat[i][j] ;
    }
  }
}

patMyMatrix::patMyMatrix(unsigned long n, unsigned long m, patReal init) :
  nr(n),
  nc(m),
  data(new patReal*[n])
{
  unsigned long i ;
  unsigned long j ;
  data[0] = new patReal[nr * nc] ;
  for (i = 1 ; i < nr ; ++i) {
    data[i] = data[i-1] + nc ;
  }
  for (i = 0 ; i < nr ; ++i) {
    for (j = 0 ; j < nc ; ++j) {
      data[i][j] = init ;
    }
  }
}

patMyMatrix::patMyMatrix(unsigned long n, unsigned long m)  :
  nr(n),
  nc(m),
  data(new patReal*[n])
{
  unsigned long i ;
  data[0] = new patReal[nr * nc] ;
  for (i = 1 ; i < nr ; ++i) {
    data[i] = data[i-1] + nc ;
  }
}

patMyMatrix::~patMyMatrix() {
  if (data != NULL) {
    delete[] (data[0]) ;
    delete[] (data) ;
  }
}


ostream& operator<<(ostream &str, const patMyMatrix& x) {
  for (unsigned long i = 0 ; i < x.nr ; ++i) {
    for (unsigned long j = 0 ; j < x.nc ; ++j) {
      str << x[i][j] ;
      if (j < x.nc-1) {
	str << '\t' ;
      }
    }
    str << endl ;
  }
  return str ;
}

void mult(const patMyMatrix& a, const patMyMatrix& b, patMyMatrix& res, patError*& err) {
  if (a.nCols() != b.nRows()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nCols() << " columns ans B has " << b.nRows() << " rows." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (a.nRows() != res.nRows()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nRows() << " rows ans RES has " << res.nRows() << " rows" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (b.nCols() != res.nCols()) {
    stringstream str ;
    str << "Incompatible sizes: B has " << a.nCols() << " columns ans RES has " << res.nCols() << " columns." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i <res.nRows() ; ++i) {
    for (unsigned long j = 0 ; j < res.nCols() ; ++j) {
      res[i][j] = 0.0 ;
      for (unsigned long k = 0 ; k < a.nCols() ; ++k) {
	res[i][j] += a[i][k] * b[k][j] ;
      }
    }
  }
}

void multATranspB(const patMyMatrix& aTransp, const patMyMatrix& b, patMyMatrix& res, patError*& err) {
  if (aTransp.nRows() != b.nRows()) {
    stringstream str ;
    str << "Incompatible sizes: A' has " << aTransp.nRows() << " rows ans B has " << b.nRows() << " rows" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (aTransp.nCols() != res.nRows()) {
    stringstream str ;
    str << "Incompatible sizes: A' has " << aTransp.nCols() << " columns and RES has " << res.nRows() << " rows" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (b.nCols() != res.nCols()) {
    stringstream str ;
    str << "Incompatible sizes: B has " << b.nCols() << " columns and RES has " << res.nCols() << " columns" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i <res.nRows() ; ++i) {
    for (unsigned long j = 0 ; j < res.nCols() ; ++j) {
      res[i][j] = 0.0 ;
      for (unsigned long k = 0 ; k < aTransp.nRows() ; ++k) {
	res[i][j] += aTransp[k][i] * b[k][j] ;
      }
    }
  }
}


void multABTransp(const patMyMatrix& a, const patMyMatrix& bTransp, patMyMatrix& res, patError*& err) {
  if (a.nCols() != bTransp.nCols()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nCols() << " columns and B' has " << bTransp.nCols() << " columns" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (a.nRows() != res.nRows()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nRows() << " rows and RES has " << res.nRows() << " rows" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (bTransp.nRows() != res.nCols()) {
    stringstream str ;
    str << "Incompatible sizes: B' has " << bTransp.nRows() << " rows and RES has " << res.nCols() << " columns" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i <res.nRows() ; ++i) {
    for (unsigned long j = 0 ; j < res.nCols() ; ++j) {
      res[i][j] = 0.0 ;
      for (unsigned long k = 0 ; k < a.nCols() ; ++k) {
	res[i][j] += a[i][k] * bTransp[j][k] ;
      }
    }
  }
}

void multVec(const patMyMatrix& a, 
	     const patVariables& b, 
	     patVariables& res,
	     patError*& err) {
  

  if (a.nCols() != b.size()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nCols() << " columns and B has " << b.size() << " elements" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (a.nRows() != res.size()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nRows() << " rows and RES has " << res.size() << " elements" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i <res.size() ; ++i) {
    res[i] = 0.0 ;
    for (unsigned long k = 0 ; k < a.nCols() ; ++k) {
      res[i] += a[i][k] * b[k] ;
    }
  }
}

void multTranspVec(const patMyMatrix& a, 
		   const patVariables& b, 
		   patVariables& res, 
		   patError*& err) {

  if (a.nRows() != b.size()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nRows() << " rows and B has " << b.size() << " elements" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (a.nCols() != res.size()) {
    stringstream str ;
    str << "Incompatible sizes: A has " << a.nCols() << " columns and RES has " << res.size() << " elements" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ; i <res.size() ; ++i) {
    res[i] = 0.0 ;
    for (unsigned long k = 0 ; k < a.nRows() ; ++k) {
      res[i] += a[k][i] * b[k] ;
    }
  }
}

patMyMatrix& patMyMatrix::operator=(const patMyMatrix& a) {
  if (this != &a) {
    unsigned long i ;
    unsigned long j ;
    if (nr != a.nr || nc != a.nc) {
      if (data != NULL) {
	delete[] (data[0]) ;
	delete[] data ;
      }
      nr = a.nr ;
      nc = a.nc ;
      data = new patReal*[nr] ;
      data[0] = new patReal[nr * nc] ;
    }
    for (i = 1 ; i < nr ; ++i) {
      data[i] = data[i-1] + nc ;
    }
    for (i = 0 ; i < nr ; ++i) {
      for (j = 0 ; j < nc ; ++j) {
	data[i][j] = a[i][j] ;
      }
    }
  }
  return *this ;
}


patVariables patMyMatrix::getRow(unsigned long r) {
  if (r >= nRows()) {
    WARNING("Out of bounds") ;
    return patVariables() ;
  }
  patVariables theRow(nCols()) ;
  for (unsigned long i = 0 ; i < nCols() ; ++i) {
    theRow[i] = data[r][i] ;
  }
  return theRow ;
}

patVariables patMyMatrix::getColumn(unsigned long r) {
  if (r >= nCols()) {
    WARNING("Out of bounds") ;
    return patVariables() ;
  }
  patVariables theColumn (nRows()) ;
  for (unsigned long i = 0 ; i < nRows() ; ++i) {
    theColumn[i] = data[i][r] ;
  }
  return theColumn ;
}
