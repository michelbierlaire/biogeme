//-*-c++-*------------------------------------------------------------
//
// File name : patZeroOneUniform.cc
// Author :    Michel Bierlaire
// Date :      Sat Jun 12 17:11:43 2010
//
//--------------------------------------------------------------------

#include "patZeroOneUniform.h"
#include "patUniform.h" 
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patFileNames.h"


patZeroOneUniform::patZeroOneUniform(patBoolean dumpDrawsOnFile) : 
  patRandomNumberGenerator(dumpDrawsOnFile),
  uniformNumberGenerator(NULL),
  logFile(NULL) {
  patError* err(NULL) ;
  if (dumpDrawsOnFile) {
    logFile = new ofstream(patFileNames::the()->getUnifDrawLogFile(err).c_str()) ;  }
}

patZeroOneUniform::~patZeroOneUniform() {
  if (logFile != NULL) {
    logFile->close() ;
    DELETE_PTR(logFile) ;
  }
}


void patZeroOneUniform::setUniform(patUniform* rng) {
  uniformNumberGenerator = rng ;
}

pair<patReal,patReal> patZeroOneUniform::getNextValue(patError*& err ) {
  if (uniformNumberGenerator == NULL) {
    err = new patErrMiscError("No pseudo-random generator specified") ;
    WARNING(err->describe());
    return pair<patReal,patReal>() ;
  }
  patReal zeroOneDraw = uniformNumberGenerator->getUniform(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return pair<patReal,patReal>() ;
  }
  pair<patReal,patReal> result(zeroOneDraw,zeroOneDraw) ;
  if (logFile != NULL) {
    *logFile << zeroOneDraw << '\t' << zeroOneDraw << endl ;
  }
  return result ;
}


patBoolean patZeroOneUniform::isSymmetric() const {
  return patFALSE ;
}
