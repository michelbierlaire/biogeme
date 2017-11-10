//-*-c++-*------------------------------------------------------------
//
// File name : patSimBasedMaxLikeOptimization.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jun 11 09:14:32 2004
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patSimBasedMaxLikeOptimization.h"
#include "patModelSpec.h"
#include "patMaxLikeProblem.h"

patSimBasedMaxLikeOptimization::
patSimBasedMaxLikeOptimization(patMaxLikeProblem* aProblem,
			       unsigned long nbrOfdraws) :
  patSimBasedOptimizationProblem(aProblem,nbrOfdraws) {

}

patSimBasedMaxLikeOptimization::~patSimBasedMaxLikeOptimization() {

}
  
void patSimBasedMaxLikeOptimization::setAlgoNbrOfDraws(unsigned long n) {
  algoControlledNbrOfDraws = n ;
  patModelSpec::the()->setAlgoNumberOfDraws(n) ;
}
