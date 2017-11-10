//-*-c++-*------------------------------------------------------------
//
// File name : bioRawResults.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Aug 17 13:24:25 2009
//
//--------------------------------------------------------------------

#ifndef bioRawResults_h
#define bioRawResults_h

class bioMinimizationProblem ;
class trFunction ;
#include "patError.h"
#include "patVariables.h"
#include "trVector.h"
#include "trHessian.h"

class bioRawResults {
 public:
  bioRawResults(bioMinimizationProblem* p, patVariables solution, patError*& err) ;
  patULong getSize() const;
  patVariables solution ;
  bioMinimizationProblem* theProblem ;
  patReal valueAtSolution ;
  trFunction* theFunction ;
  trVector gradient ;
  patReal gradientNorm ;
  patMyMatrix* varianceCovariance ;
  patMyMatrix* robustVarianceCovariance ;
  //  trHessian hessian ;
  map<patReal,patVariables> eigenVectors ;
  patReal smallestSingularValue ;

 private:
  void compute(patError*& err) ;
};

#endif
