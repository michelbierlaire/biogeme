//-*-c++-*------------------------------------------------------------
//
// File name : patMtl.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed Mar  6 15:40:33 2002
//
// This file contains the interface to the Matrix Template Library 
//--------------------------------------------------------------------

#ifndef patMtl_h
#define patMtl_h

#include "mtl/matrix.h"
#include "mtl/mtl.h"
#include "mtl/utils.h"
#include "mtl/linalg_vec.h"
#include "mtl/lu.h"
#include "patType.h"

using namespace mtl;

/**
 */
typedef dense1D<patReal> mtlVector;

/**
 */
typedef matrix<patReal,
    rectangle<>, 
    compressed<>,
    row_major >::type mtlSparseMatrix;

/**
 */
typedef matrix<patReal, rectangle<>, dense<>, row_major>::type mtlDenseMatrix;

/**
 */
ostream& operator<<(ostream &str, const mtlVector& x) ;


#endif
