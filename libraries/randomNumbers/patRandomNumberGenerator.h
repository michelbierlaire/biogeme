//-*-c++-*------------------------------------------------------------
//
// File name : patRandomNumberGenerator.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Aug 24 15:19:49 2003
//
//--------------------------------------------------------------------

#ifndef patRandomNumberGenerator_h
#define patRandomNumberGenerator_h

#include "patConst.h"

class patError ;

/**
   @doc Virtual class for random number generators
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Sun Aug 24 15:19:49 2003)
 */

class patRandomNumberGenerator {

public:
  patRandomNumberGenerator(patBoolean dumpDrawsOnFile) ;
  virtual ~patRandomNumberGenerator() {} ;
  // REturns the next draw, and the zero-one draw that was used to generate it.
  virtual pair<patReal,patReal> getNextValue(patError*& err) = PURE_VIRTUAL ;
  patReal getZeroOneDraw(patError*& err)  ;
  virtual patBoolean isSymmetric() const = PURE_VIRTUAL ;
  virtual patBoolean isNormal() const ;
protected:
  patBoolean dumpDrawsOnFile ;
} ;

#endif

