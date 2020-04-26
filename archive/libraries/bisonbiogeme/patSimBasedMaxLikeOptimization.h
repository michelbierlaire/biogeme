//-*-c++-*------------------------------------------------------------
//
// File name : patSimBasedMaxLikeOptimization.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Fri Jun 11 09:07:47 2004
//
//--------------------------------------------------------------------

#ifndef patSimBasedMaxLikeOptimization_h
#define patSimBasedMaxLikeOptimization_h

#include "patSimBasedOptimizationProblem.h"

class patMaxLikeProblem ;

class patSimBasedMaxLikeOptimization : public patSimBasedOptimizationProblem {
 
 public:
  patSimBasedMaxLikeOptimization(patMaxLikeProblem* aProblem,
				 unsigned long nbrOfdraws) ;

  virtual ~patSimBasedMaxLikeOptimization() ;
  
  virtual void setAlgoNbrOfDraws(unsigned long n) ;

};

#endif 
