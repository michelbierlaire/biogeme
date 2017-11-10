//-*-c++-*------------------------------------------------------------
//
// File name : patUnixUniform.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 16:13:29 2003
//
//--------------------------------------------------------------------

#ifndef patUnixUniform_h
#define patUnixUniform_h


/**
   @doc Uniformly distributed random numbers using the Unix function
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Thu Mar  6 16:04:50 2003)
 */

#include "patUniform.h"

class patUnixUniform : public patUniform {

 public:

  patUnixUniform() ;

  /**
     @param s seed for the pseud-random number generator
   */
  patUnixUniform(unsigned long s) ;

  /**
     Set the seed value for the pseudo-random number generator
   */

  void setSeed(unsigned long s) ;
  /**
     Set the seed value for the pseudo-random number generator
   */

  unsigned long getSeed() ;

  /**
     Return random numbers uniformly distributed between 0 and 1
   */
  patReal getUniform(patError*& err) ;

  patString getType() const ;

 private:
  unsigned long seed ;

};

#endif
