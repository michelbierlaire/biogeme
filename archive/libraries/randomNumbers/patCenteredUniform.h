//-*-c++-*------------------------------------------------------------
//
// File name : patCenteredUniform.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Jan  6 17:53:40 2004
//
//--------------------------------------------------------------------

#ifndef patCenteredUniform_h
#define patCenteredUniform_h



/**
   @doc Generate pseudo-random numbers from a unifrom distribution [-1:1].
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Tue Jan  6 17:53:40 2004)
 */

#include <fstream>
#include "patType.h"
#include "patRandomNumberGenerator.h"
class patError ;
class patUniform ;

class patCenteredUniform : public patRandomNumberGenerator {

 public:
  /**
   */
  patCenteredUniform(patBoolean dumpDrawsOnFile) ;
  /**
   */
  virtual ~patCenteredUniform() ;
  /**
   */
  void setUniform(patUniform* rng) ;

  /**
   */
  virtual pair<patReal,patReal> getNextValue(patError*& err) ;

  /**
   */
  virtual patBoolean isSymmetric() const  ;

 private:
  patUniform* uniformNumberGenerator ;
  ofstream* logFile ;
};
#endif
