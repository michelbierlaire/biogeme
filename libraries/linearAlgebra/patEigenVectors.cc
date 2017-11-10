//-*-c++-*------------------------------------------------------------
//
// File name : patEigenVectors.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jan 20 14:55:54 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patEigenVectors.h"
#include "patMyMatrix.h"
#include "patHybridMatrix.h"
#include "patSvd.h"

patEigenVectors::patEigenVectors(patHybridMatrix* aMatrix,
				 int svdMaxIter, 
				 patError*& err) : theMatrix(NULL),
						   theSvd(NULL) {
  if (aMatrix != NULL) {
    theMatrix = new patMyMatrix(*aMatrix,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  theSvd = new patSvd(theMatrix) ;
  theSvd->computeSvd(svdMaxIter, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

patEigenVectors::~patEigenVectors() {
  DELETE_PTR(theSvd) ;
  DELETE_PTR(theMatrix) ;
}
map<patReal,patVariables> patEigenVectors::getEigenVector(patReal threshold) {
  if (theSvd == NULL) {
    return map<patReal,patVariables>() ;
  }
  return theSvd->getEigenVectorsOfZeroEigenValues(threshold) ;
}

unsigned int patEigenVectors::getRows() {
  if (theMatrix == NULL) {
    return 0 ;
  }
  return theMatrix->nRows() ;
}

unsigned int patEigenVectors::getCols() {
  if (theMatrix == NULL) {
    return 0 ;
  }
  return theMatrix->nCols() ;
}
