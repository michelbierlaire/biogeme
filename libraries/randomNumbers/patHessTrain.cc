//-*-c++-*------------------------------------------------------------
//
// File name : patHessTrain.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sat Apr  3 16:52:08 2004
//
//--------------------------------------------------------------------

#include <sstream>
#include <vector>
#include <algorithm>
#include "patDisplay.h"
#include "patShuffleVector.h"
#include "patHessTrain.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patUnixUniform.h"

patHessTrain::patHessTrain(unsigned long nDraws,
			   patUnixUniform* aRng) : 
  numbers(nDraws),
  theRandomGenerator(aRng),
  currentIndex(0) {
  
 }

patReal patHessTrain::getUniform(patError*& err) {
  if (currentIndex == 0) {
    generateStructuredNoise(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
  }
  patReal theNumber = numbers[currentIndex] ;
  ++currentIndex ;
  if (currentIndex >= numbers.size()) {
    currentIndex = 0 ;
  }
  return theNumber ;
}

void patHessTrain::generateStructuredNoise(patError*& err) {
  
  unsigned long nDraws = numbers.size() ;
  for (unsigned long i = 0 ; i < nDraws ; ++i) {
    numbers[i] = patReal(i) / patReal(nDraws) ;
  }
  
  patReal k = 0.0 ;
  while ((k == 0.0) || (k == 1.0)) {
    k = theRandomGenerator->getUniform(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
  }
  k /= patReal(nDraws) ;
  for (unsigned long i = 0 ; i < nDraws ; ++i) {
    numbers[i] += k ;
  }
    
  patShuffleVector<patReal>()(&numbers) ;

  return ;
}

patString patHessTrain::getType() const {
  return patString("Modified Latin Hypercube Sampling") ;
}
