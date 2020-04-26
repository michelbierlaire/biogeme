//-*-c++-*------------------------------------------------------------
//
// File name : patSimBasedOptimizationProblem.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jun 11 08:55:31 2004
//
//--------------------------------------------------------------------
#ifndef patSimBasedOptimizationProblem_h
#define patSimBasedOptimizationProblem_h

#include "patConst.h"

class patNonLinearProblem ;

/**
@doc Nonlinear optimization problem requiring Monte-Carlo simulation
to be evaluated. Thjis object enables the algorithms to control the
number of draws from one iteration to the next, in order to speed up
the optimization process. 

   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Fri Jun 11 08:55:31 2004)

 */

class patSimBasedOptimizationProblem {

 public:
  
  /**
     Ctor
     @param aProblem pointer to a non linear problem to solve
     @param nbrOfdraws required number of draws to fully evaluate the problem
   */
  patSimBasedOptimizationProblem(patNonLinearProblem* aProblem,
				 unsigned long nbrOfdraws) ;
  
  /**
     @return Number of draws necessary to fully evaluate the problem,
     according to the user
   */
  unsigned long getUserNbrOfDraws() const ;
  
  /**
     @return Number of draws specified by the algorithm
   */
  unsigned long getAlgoNbrOfDraws() const ;

  /**
     Set the number of draws desired by the algorithm
   */
  virtual void setAlgoNbrOfDraws(unsigned long n) = PURE_VIRTUAL ;

 public:

  patNonLinearProblem* theProblem ;

  
 protected:
  unsigned long userRequiredNbrOfDraws ;
  unsigned long algoControlledNbrOfDraws ;
};
#endif
