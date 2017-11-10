//-*-c++-*------------------------------------------------------------
//
// File name : patBinomial.cc
// Author :    \URL[Michel Bierlaire]{http://transp-or.epfl.ch}
// Date :      Sun May 11 16:01:06 2008
//
//--------------------------------------------------------------------

#include "patUniform.h"
#include "patBinomial.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patBinomial::patBinomial(patULong _n, patReal _p, patUniform* rng) :
  n(_n),
  p(_p),
  uniformNumberGenerator(rng) {

}

patBinomial::~patBinomial() {

}


patULong patBinomial::getNextValue(patError*& err) {
  if (uniformNumberGenerator == NULL) {
    err = new patErrNullPointer("patUniform") ;
    WARNING(err->describe()) ;
    return patULong() ;
  }
  patULong count = 0 ;
  for (patULong i = 0 ; i < n ; ++i) {
    patReal draw = uniformNumberGenerator->getUniform(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patULong() ;
    }
    if (draw <= p) {
      ++count ;
    }
  }
  return count ;
}

void patBinomial::setUniform(patUniform* rng) {
  uniformNumberGenerator = rng ;
}
