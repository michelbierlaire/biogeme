//-*-c++-*------------------------------------------------------------
//
// File name : patHessTrain.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sat Apr  3 16:51:17 2004
//
//--------------------------------------------------------------------

#ifndef patHessTrain_h
#define patHessTrain_h

#include "patUniform.h"
#include "patError.h"
#include "patVariables.h"

class patUnixUniform ;

/**
   Generate draws from the Hess-Train distribution
 */
class patHessTrain : public patUniform {
 public:
  patHessTrain(unsigned long nDraws,
	       patUnixUniform* aRng) ;
  patReal getUniform(patError*& err);
  patString getType() const ;
  
 private:

  /**
     Generate a vector of shuffled structured noise of length nDraws
  */
  void generateStructuredNoise(patError*& err) ;

private:

  /**
   */
  patVariables numbers ;

  /**
   */
  patUnixUniform* theRandomGenerator ;

  /**
   */
  unsigned long currentIndex ;



};


#endif

