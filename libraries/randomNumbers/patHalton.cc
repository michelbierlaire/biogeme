//-*-c++-*------------------------------------------------------------
//
// File name : patHalton.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Sep  1 21:48:16 2003
//
//--------------------------------------------------------------------

#include <sstream>
#include <vector>
#include "patDisplay.h"
#include "patHalton.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"

patHalton::patHalton(unsigned int nSeries, 
		     patULong maxPrimeNumber,
		     patULong draws,
		     patError*& err) :
  primes (maxPrimeNumber), 
  drawsPerSerie(draws),
  currentNumber(nSeries,1),
  currentIndex(0) {
  if (primes.getNumberOfPrimes() < nSeries) {
    stringstream str ;
    str << nSeries << " Halton series must be generated, but there are only " << primes.getNumberOfPrimes() << " prime numbers available" << endl;
    str << "Increase the value of gevMaxPrimeNumber in the parameters file" << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
}

patReal patHalton::getUniform(patError*& err) {
  
  if (currentIndex >= currentNumber.size()) {
    currentIndex = 0 ;
  }
  if (currentNumber[currentIndex] > drawsPerSerie) {
    ++currentIndex ;
  }

  unsigned int base = primes.getPrime(currentIndex,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal result = computeHalton(base,currentNumber[currentIndex]) ;
  ++(currentNumber[currentIndex]) ;
  return result ;
}

patReal patHalton::computeHalton(unsigned int basePrime, 
				 unsigned int nbr) {

  vector<unsigned int> decomposition ;
  unsigned long ratio = nbr ;
  while (ratio != 0) {
    unsigned int modulo = ratio % basePrime ;
    decomposition.push_back(modulo) ;
    ratio /= basePrime ;
  }
  patReal hh(0.0) ;
  patReal exponent(-1.0) ;
  for (vector<unsigned int>::iterator iter = decomposition.begin() ;
       iter != decomposition.end() ;
       ++iter) {
    hh +=  (*iter) * pow(basePrime,exponent) ;
    exponent -= 1.0 ;
  }

//   if (nbr <= 20) {
//     DEBUG_MESSAGE("Halton("<< basePrime << "," << nbr << ")=" << hh) ;
//   }

  return hh ;
}

patString patHalton::getType() const {
  return patString("Halton") ;
}
