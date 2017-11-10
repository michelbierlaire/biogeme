//-*-c++-*------------------------------------------------------------
//
// File name : patBinomial.h
// Author :    \URL[Michel Bierlaire]{http://transp-or.epfl.ch}
// Date :      Sun May 11 15:55:07 2008
//
//--------------------------------------------------------------------

#ifndef patBinomial_h
#define patBinomial_h

/**
   @doc Generate pseudo-random numbers from a binomial distribution B(n,p). 
   @author \URL[Michel Bierlaire]{http://transp-or.epfl.ch}, EPFL (Sun May 11 15:55:07 2008)
 */

#include "patRandomNumberGenerator.h"
#include "patType.h"
class patError ;
class patUniform ;

class patBinomial {

 public:

  /**
   */
  patBinomial(patULong _n, patReal _p, patUniform* rng = NULL) ;

  /**
   */
  virtual ~patBinomial() ;

  /**
   */
  patULong getNextValue(patError*& err) ;

  /**
   */
  void setUniform(patUniform* rng) ;

 private :
  patULong n ;
  patReal p ;
  patUniform* uniformNumberGenerator ;
};

#endif
