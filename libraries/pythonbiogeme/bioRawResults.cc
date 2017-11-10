//-*-c++-*------------------------------------------------------------
//
// File name : bioRawResults.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Aug 17 13:32:06 2009
//
//--------------------------------------------------------------------

#include "patErrNullPointer.h"
#include "bioRawResults.h"
#include "bioMinimizationProblem.h"
#include "trFunction.h"

bioRawResults::bioRawResults(bioMinimizationProblem* p, 
			     patVariables s, 
			     patError*& err): 
  solution(s), 
  theProblem(p), 
  gradient(s.size()), 
  varianceCovariance(NULL),
  robustVarianceCovariance(NULL),
  smallestSingularValue(0.0) {
  if (theProblem == NULL) {
    err = new patErrNullPointer("bioMinimizationProblem") ;
    WARNING(err->describe()) ;
    return ;
  }
  theFunction = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patULong varCovarSize = theProblem->getSizeOfVarCovar() ;
  varianceCovariance = new patMyMatrix(varCovarSize,varCovarSize) ; 
  patULong dim = theFunction->getDimension() ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  robustVarianceCovariance = new patMyMatrix(dim,dim) ; 
  compute(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

void bioRawResults::compute(patError*& err) {

  patBoolean success ;

  valueAtSolution = -theFunction->computeFunctionAndDerivatives(&solution,
								&gradient,
								NULL,
								&success,
								err) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  DEBUG_MESSAGE("Final loglikelihood: " << valueAtSolution) ;
  if (!success) {
    err = new patErrMiscError("Unable to compute the function or gradient at the solution.") ;
    WARNING(err->describe()) ;
    return ;
  }

  gradientNorm = 0.0 ;
  for (trVector::iterator i = gradient.begin() ;
       i != gradient.end() ;
       ++i) {
    gradientNorm += (*i) * (*i) ;
  }
  gradientNorm = sqrt(gradientNorm) ;

  success = theProblem->computeVarCovar(&solution,
					varianceCovariance,
					robustVarianceCovariance,
					&eigenVectors,
					&smallestSingularValue,
					err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (!success) {
    WARNING("Unable to compute the variance-covariance at the solution.") ;
  }


}

patULong bioRawResults::getSize() const {
  return solution.size() ;
}


