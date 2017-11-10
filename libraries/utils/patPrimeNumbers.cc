//-*-c++-*------------------------------------------------------------
//
// File name : patPrimeNumbers.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Sep  1 20:51:51 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patDisplay.h"
#include "patPrimeNumbers.h"
#include "patErrMiscError.h"

patPrimeNumbers::patPrimeNumbers(unsigned long upperBound) :  primesComputed(patFALSE),
							      up(upperBound), 
							      mywork(upperBound+1) {

}
unsigned long patPrimeNumbers::getNumberOfPrimes() {
  if (!primesComputed) {
    computePrimes() ;
  }
  return myprimes.size() ;
}

unsigned long patPrimeNumbers::getPrime(unsigned long i, patError*& err ) {
  if (!primesComputed) {
    computePrimes() ;
  }
  if (i < myprimes.size()) {
    return myprimes[i] ;
  }
  else {
    stringstream str ;
    str << "There are only " << getNumberOfPrimes() << " primes available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return 0 ;
  }
}

void patPrimeNumbers::computePrimes() {
  for (unsigned long i = 1 ; i <= up ; ++i) {
    mywork[i] = i ;
  } 
  unsigned long max = int(ceil(sqrt(patReal(up)))) ;

  for (unsigned long i = 2 ; i <= max ; ++i) {
    if (mywork[i] != 0) {
      unsigned long mult = i ;
      while (mult <= up) {
	mult += i ;
	if (mult <= up) {
	  mywork[mult] = 0 ;
	}
      }
    }
  }

  for (unsigned long i = 1 ; i <= up ; ++i) {
    if (mywork[i] != 0 && mywork[i] != 1) {
      myprimes.push_back(i) ;
    }
  } 
  primesComputed = patTRUE ;
}
