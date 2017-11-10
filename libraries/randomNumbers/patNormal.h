//-*-c++-*------------------------------------------------------------
//
// File name : patNormal.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 16:54:05 2003
//
//--------------------------------------------------------------------

#ifndef patNormal_h
#define patNormal_h



/**
   @doc Generate pseudo-random numbers from a normal distribution N(0,1). This class is a singleton. 
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Thu Mar  6 16:54:05 2003)
 */

#include <fstream>
#include "patType.h"
#include "patRandomNumberGenerator.h"
class patError ;
class patUniform ;

class patNormal : public patRandomNumberGenerator {

 public:
  /**
   */
  patNormal(patBoolean dumpDrawsOnFile) ;
  /**
   */
  virtual ~patNormal() ;
  /**
   */
  void setUniform(patUniform* rng) ;

  /**
   */
  virtual pair<patReal,patReal> getNextValue(patError*& err) ;

  virtual patBoolean isSymmetric() const ;

  virtual patBoolean isNormal() const ;

 private:
  patBoolean first ;
  patUniform* uniformNumberGenerator ;
  ofstream* logFile ;
  patReal v1 ;
  patReal v2 ;
};
#endif
