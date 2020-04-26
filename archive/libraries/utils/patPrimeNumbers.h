//-*-c++-*------------------------------------------------------------
//
// File name : patPrimeNumbers.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Sep  1 20:47:56 2003
//
//--------------------------------------------------------------------

#ifndef patPrimeNumbers_h
#define patPrimeNumbers_h

#include "patError.h"
#include "patConst.h"
#include <vector>
/**
   @doc Class generating prime numbers
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Mon Sep  1 20:47:56 2003)
   
 */

class patPrimeNumbers {

 public:
  patPrimeNumbers(unsigned long upperBound) ;
  unsigned long getNumberOfPrimes() ;
  unsigned long getPrime(unsigned long i, patError*& err ) ;

 protected:
  void computePrimes() ;
 private:

  patBoolean primesComputed ;
  unsigned long up ;
  vector<unsigned long> myprimes ;
  vector<unsigned long> mywork ;

};

#endif
