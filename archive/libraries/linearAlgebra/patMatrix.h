//-*-c++-*------------------------------------------------------------
//
// File name : patMatrix.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Jun 15 18:04:59 1999
//
//--------------------------------------------------------------------

#ifndef patMatrix_h
#define patMatrix_h

#include <vector>
#include "patError.h"
#include "patType.h"
#include "patVariables.h"

/**
 */
typedef vector<patReal > patMatrixRow ;
/**
 */
typedef vector<patMatrixRow > patMatrix ;

/**
 */
patMatrix operator*(const patMatrix& A, const patMatrix& B) ;
/**
 */
patVariables operator*(const patMatrix& A, const patVariables& x) ;
/**
 */
patVariables transpMult(const patMatrix& A, const patVariables& x) ;


/**
 */
patMatrix transpose(const patMatrix& A, 
		    patError*& err) ;
/**
 */
ostream& operator<<(ostream &str, const patMatrix& x) ;

#endif

