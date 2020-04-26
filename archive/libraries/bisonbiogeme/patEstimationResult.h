//-*-c++-*------------------------------------------------------------
//
// File name : patEstimationResult.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Wed May 16 21:34:56 2001
//
//--------------------------------------------------------------------

#ifndef patEstimationResult_h
#define patEstimationResult_h

#include "patConst.h"
#include "patVariables.h"
//#include "patMtl.h"
#include "patAbsTime.h"

#include "patMyMatrix.h"

// LAPACK++
//  #include "lafnames.h"
//  #include "lapack.h"
//  #include "symd.h"

/**
  @doc Contains all information needed to produce the results
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Wed May 16 21:34:56 2001) 
 */
class patEstimationResult {
public:
  patEstimationResult() ;
  void computeCorrelation() ;
  patReal nullLoglikelihood ;
  patReal initLoglikelihood ;
  patReal cteLikelihood ;
  patULong iterations ;
  patString diagnostic ;
  patReal loglikelihood ;
  patReal gradientNorm ;
  //  LaGenMatDouble* varCovarMatrix ;
  patBoolean isVarCovarAvailable ;
  patMyMatrix* varCovarMatrix ;
  patMyMatrix* robustVarCovarMatrix ;
  patMyMatrix* correlation ;
  patMyMatrix* robustCorrelation ;
  patBoolean isRobustVarCovarAvailable ;
  unsigned long numberOfObservations ;
  unsigned long numberOfIndividuals ;
  map<patReal,patVariables> eigenVectors ;
  patReal smallestSingularValue ;
  patString runTime ;
  patBoolean halton ;
  patBoolean hessTrain ;
};

#endif
