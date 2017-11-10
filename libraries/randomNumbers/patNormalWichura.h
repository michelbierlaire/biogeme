//-*-c++-*------------------------------------------------------------
//
// File name : patNormalWichura.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Wed May 12 14:54:28 2004
//
//--------------------------------------------------------------------

#ifndef patNormalWichura_h
#define patNormalWichura_h



/**
   @doc Generate pseudo-random numbers from a normal distribution
   N(0,1) using the Algorithm AS241 Appl. Statist. (1988) Vol. 37,
   No. 3, which  produces the normal deviate z corresponding to a
   given lower tail area of p; z is accurate to about 1 part in 10**16.
   @author \URL[Michel  Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Wed May 12 14:54:28 2004)
 */

#include <fstream>
#include "patType.h"
#include "patRandomNumberGenerator.h"
class patError ;
class patUniform ;

class patNormalWichura : public patRandomNumberGenerator {

 public:
  /**
   */
  patNormalWichura(patBoolean dumpDrawsOnFile = patFALSE) ;

  /**
   */
  virtual ~patNormalWichura() ;
  /**
   */
  void setUniform(patUniform* rng) ;

  /**
   */
  virtual pair<patReal,patReal> getNextValue(patError*& err) ;

  virtual patBoolean isSymmetric() const ;

  virtual patBoolean isNormal() const ;

 private:
  patUniform* uniformNumberGenerator ;
  ofstream* logFile ;
  patReal zero, one, half ;
  patReal split1, split2 ;
  patReal const1, const2 ;
  patReal a0,a1,a2,a3,a4,a5,a6,a7 ;
  patReal    b1,b2,b3,b4,b5,b6,b7 ;
  patReal c0,c1,c2,c3,c4,c5,c6,c7 ;
  patReal    d1,d2,d3,d4,d5,d6,d7 ;
  patReal e0,e1,e2,e3,e4,e5,e6,e7 ;
  patReal    f1,f2,f3,f4,f5,f6,f7 ;
};
#endif
