//-*-c++-*------------------------------------------------------------
//
// File name : patHalton.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Sep  1 21:44:40 2003
//
//--------------------------------------------------------------------

#ifndef patHalton_h
#define patHalton_h

#include "patUniform.h"
#include "patPrimeNumbers.h"

/**
   Generate Halton draws
 */
class patHalton : public patUniform {
 public:
  patHalton(unsigned int nSeries,
	    patULong maxPrimeNumber,
	    patULong draws,
	    patError*& err) ;
  patReal getUniform(patError*& err);
  patString getType() const ;
  
 private:

  patReal computeHalton(unsigned int base, unsigned int number) ;

  patPrimeNumbers primes ;
  patULong drawsPerSerie ;
  vector<unsigned int> currentNumber ;
  unsigned long currentIndex ;
};


#endif

