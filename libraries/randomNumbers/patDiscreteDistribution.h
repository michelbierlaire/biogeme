//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteDistribution.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Mar  6 16:32:31 2003
//
//--------------------------------------------------------------------

#ifndef patDiscreteDistribution_h
#define patDiscreteDistribution_h

#include <vector>
#include "patType.h"

class patUniform ;

/**
   @doc This objects generates draw from a discrete
   distribution. Based on a table of real values representing relative
   weights, it returns a random index, such that the probability of
   drawing j is the weight of j divided by the sum of all weights in
   the table.
 */

class patDiscreteDistribution {

 public:

  // normalized is set to patTRUE is the entries of the vector d sum up to one.
  patDiscreteDistribution(const vector<patReal>* d,
			  patUniform* rng,
			  patBoolean normalized = patFALSE) ;
  pair<vector<patReal>::size_type,patReal> getDrawWithProba() ;
  vector<patReal>::size_type operator()() ;

 private:

  const vector<patReal>* weights ;
  patUniform* randomNumberGenerator ;
  patBoolean normalized ;
  
};
#endif
