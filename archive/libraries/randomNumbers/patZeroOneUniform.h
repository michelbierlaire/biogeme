//-*-c++-*------------------------------------------------------------
//
// File name : patZeroOneUniform.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sat Jun 12 17:11:05 2010
//
//--------------------------------------------------------------------

#ifndef patZeroOneUniform_h
#define patZeroOneUniform_h



/**
   @doc Generate pseudo-random numbers from a unifrom distribution [0:1].
 */

#include <fstream>
#include "patType.h"
#include "patRandomNumberGenerator.h"
class patError ;
class patUniform ;

class patZeroOneUniform : public patRandomNumberGenerator {

 public:
  /**
   */
  patZeroOneUniform(patBoolean dumpDrawsOnFile) ;
  /**
   */
  virtual ~patZeroOneUniform() ;
  /**
   */
  void setUniform(patUniform* rng) ;

  /**
   */
  virtual pair<patReal,patReal> getNextValue(patError*& err) ;

  /**
   */
  virtual patBoolean isSymmetric() const ;

 private:
  patUniform* uniformNumberGenerator ;
  ofstream* logFile ;
};
#endif
