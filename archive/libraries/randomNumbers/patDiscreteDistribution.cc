//-*-c++-*------------------------------------------------------------
//
// File name : patDiscreteDistribution.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Thu Mar  6 16:37:30 2003
//
//--------------------------------------------------------------------

#include <numeric>
#include "patDisplay.h"
#include "patConst.h"
#include "patDiscreteDistribution.h"
#include "patUniform.h"

patDiscreteDistribution::patDiscreteDistribution(const vector<patReal>* d,
						 patUniform* rng,
						 patBoolean norm) : weights(d), randomNumberGenerator(rng) , normalized(norm) {

}

vector<patReal>::size_type patDiscreteDistribution::operator()() {
  pair<vector<patReal>::size_type,patReal> r = getDrawWithProba() ;
  return r.first ;
}

pair<vector<patReal>::size_type,patReal> patDiscreteDistribution::getDrawWithProba() {
  if (randomNumberGenerator == NULL || weights == NULL) {
    WARNING("Null pointer.") ;
    return pair<vector<patReal>::size_type,patReal>(patBadId,0) ;
  }

  if (weights->empty()) {
    WARNING("No mass value") ;
    return pair<vector<patReal>::size_type,patReal>(patBadId,0) ;
  }
  pair<vector<patReal>::size_type,patReal> result ;
  patReal sum ;
  if (!normalized) {
    sum = accumulate(weights->begin(),weights->end(),0.0) ;
  }

  patError* err(NULL) ;
  patReal randomNbr = randomNumberGenerator->getUniform(err) ;
  patReal cumul = 0.0 ;
  for (vector<patReal>::size_type i = 0 ;
       i < weights->size() ;
       ++i) {
    if (normalized) {
      cumul += (*weights)[i] ;
    }
    else {
      cumul += (*weights)[i]/sum ;
    }
    if (randomNbr < cumul) {
      result.first = i ;
      if (normalized) {
	result.second = (*weights)[i] ;
      }
      else {
	result.second = (*weights)[i]/sum ;
      }
      return result ;
    }
  }

  result.first = weights->size() ;
  if (normalized) {
    result.second =  (*weights)[result.first] ;
  } 
  else {
    result.second =  (*weights)[result.first]/sum ;
  }
  return result ;
    
  
}
