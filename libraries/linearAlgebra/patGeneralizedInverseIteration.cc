//-*-c++-*------------------------------------------------------------
//
// File name : patGeneralizedInverseIteration.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jan 20 13:00:06 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patGeneralizedInverseIteration.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patLu.h"

patGeneralizedInverseIteration::
patGeneralizedInverseIteration(const patMyMatrix* aMatrix, 
			       patMyMatrix* initialEigenVectors,
			       patError*& err) :
  theMatrix(aMatrix), 
  theResult(initialEigenVectors),
  theLuDecomposition(NULL) {
  if (aMatrix == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (initialEigenVectors == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (aMatrix->nRows() != aMatrix->nCols()) {
    stringstream str ;
    str << "Not a square matrix: " 
	<< aMatrix->nRows() 
	<< "x" 
	<< aMatrix->nCols() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (aMatrix->nRows() != initialEigenVectors->nRows()) {
    stringstream str ;
    str << "Incompatible sizes: " 
	<< aMatrix->nRows() 
	<< " and " 
	<< initialEigenVectors->nRows() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (initialEigenVectors->nCols() > initialEigenVectors->nRows()) {
    stringstream str ;
    str << "Cannot ask for " 
	<< initialEigenVectors->nCols() 
	<< " eigenvector in dimension " 
	<< initialEigenVectors->nRows() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }  
}

patGeneralizedInverseIteration::~patGeneralizedInverseIteration() {
  DELETE_PTR(theLuDecomposition) ;
}

void patGeneralizedInverseIteration::computeShiftedLu(patError*& err) {
  Lu = *theMatrix ;
  for (unsigned long i = 0 ; i < theMatrix->nRows() ; ++i) {
    Lu[i][i] -= 1.0e-10 ;
  }
  theLuDecomposition = new patLu(&Lu) ;
  theLuDecomposition->computeLu(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
}
  
patMyMatrix* patGeneralizedInverseIteration::computeEigenVectors(patULong maxIter,
								 patError*& err) {
  computeShiftedLu(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  unsigned long iter = 0  ;
  while (iter < maxIter) {
    ++iter ;
    patReal normDiff = performOneIteration(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (normDiff < 1.0e-4) {
      return theResult ;
    }
  }
  return theResult ;
}
  
patReal patGeneralizedInverseIteration::performOneIteration(patError*& err) {
  patReal normDiff(0.0) ;
  for (unsigned long i = 0 ; i < theResult->nCols() ; ++i) {

    patVariables oldColumn ;
    patVariables newColumn ;
    oldColumn = newColumn = theResult->getColumn(i) ;

    newColumn = *(theLuDecomposition->solve(oldColumn,err)) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    for (unsigned long j = 0 ; j < oldColumn.size() ; ++j) {
      normDiff += 
	(oldColumn[j] - newColumn[j]) * 
	(oldColumn[j] - newColumn[j]) ;
    }
  }
  return sqrt(normDiff) ;
}


