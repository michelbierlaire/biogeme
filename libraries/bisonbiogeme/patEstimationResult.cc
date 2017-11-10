//-*-c++-*------------------------------------------------------------
//
// File name : patEstimationResult.cc
// Author :    \URL[Michel Bierlaire]{http://transp-or.epfl.ch}
// Date :      Tue Oct 28 14:16:32 2008
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patEstimationResult.h" 

patEstimationResult::patEstimationResult() : 
  isVarCovarAvailable(patFALSE),
  varCovarMatrix(NULL) ,
  robustVarCovarMatrix(NULL),
  correlation(NULL),
  robustCorrelation(NULL),
  runTime("") {
  
}

void patEstimationResult::computeCorrelation() {

  if (varCovarMatrix != NULL) {
    unsigned long n = varCovarMatrix->nRows() ;
    correlation = new patMyMatrix(n,n) ;
    for (unsigned long i = 0 ; i < n ; ++i) {
      for (unsigned long j = 0 ; j < n ; ++j) {
	(*correlation)[i][j] = (*varCovarMatrix)[i][j] / 
	  sqrt((*varCovarMatrix)[i][i] * (*varCovarMatrix)[j][j]) ;
      }
    }
  }

  if (robustVarCovarMatrix != NULL) {
    unsigned long n = robustVarCovarMatrix->nRows() ;
    robustCorrelation = new patMyMatrix(n,n) ;
    for (unsigned long i = 0 ; i < n ; ++i) {
      for (unsigned long j = 0 ; j < n ; ++j) {
	(*robustCorrelation)[i][j] = (*robustVarCovarMatrix)[i][j] / 
	  sqrt((*robustVarCovarMatrix)[i][i] * (*robustVarCovarMatrix)[j][j]) ;
      }
    }
  }

}
