//-*-c++-*------------------------------------------------------------
//
// File name : patSimBasedOptimizationProblem.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jun 11 09:10:36 2004
//
//--------------------------------------------------------------------

#include "patSimBasedOptimizationProblem.h"

patSimBasedOptimizationProblem::
patSimBasedOptimizationProblem(patNonLinearProblem* aProblem,
			       unsigned long nbrOfdraws) :
  theProblem(aProblem),
  userRequiredNbrOfDraws(nbrOfdraws),
  algoControlledNbrOfDraws(nbrOfdraws) {

}
  
unsigned long patSimBasedOptimizationProblem::getUserNbrOfDraws() const {
  return userRequiredNbrOfDraws ;
}
  
unsigned long patSimBasedOptimizationProblem::getAlgoNbrOfDraws() const {
  return algoControlledNbrOfDraws ;
}


