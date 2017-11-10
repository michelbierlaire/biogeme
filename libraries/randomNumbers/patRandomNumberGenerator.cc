//-*-c++-*------------------------------------------------------------
//
// File name : patRandomNumberGenerator.cc
// Author :    Michel Bierlaire
// Date :      Fri Oct 22 15:34:43 2010
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patRandomNumberGenerator.h"
#include "patErrMiscError.h"

patRandomNumberGenerator::patRandomNumberGenerator(patBoolean d) :
  dumpDrawsOnFile(d) {
}

patBoolean patRandomNumberGenerator::isNormal() const {
  return patFALSE ;
}

patReal patRandomNumberGenerator::getZeroOneDraw(patError*& err) {
  err = new patErrMiscError("Should not be used anymore") ;
  WARNING(err->describe()) ;
  return patReal() ;
}
