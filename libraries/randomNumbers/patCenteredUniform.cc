//-*-c++-*------------------------------------------------------------
//
// File name : patCenteredUniform.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Jan  6 17:54:55 2004
//
//--------------------------------------------------------------------

#include "patCenteredUniform.h"
#include "patUniform.h" 
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patFileNames.h"


patCenteredUniform::patCenteredUniform(patBoolean dumpDrawsOnFile) : 
  patRandomNumberGenerator(dumpDrawsOnFile),
  uniformNumberGenerator(NULL),
  logFile(NULL){
  patError* err(NULL) ;
  if (dumpDrawsOnFile) {
    logFile = new ofstream(patFileNames::the()->getUnifDrawLogFile(err).c_str()) ;  }
}

patCenteredUniform::~patCenteredUniform() {
  if (logFile != NULL) {
    logFile->close() ;
    DELETE_PTR(logFile) ;
  }
}


void patCenteredUniform::setUniform(patUniform* rng) {
  uniformNumberGenerator = rng ;
}

pair<patReal,patReal> patCenteredUniform::getNextValue(patError*& err ) {
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
  patReal result = 2.0 * zeroOneDraw - 1.0 ;
  if (logFile != NULL) {
    *logFile << result << endl ;
  }
  return pair<patReal,patReal>(result,zeroOneDraw) ;
}


patBoolean patCenteredUniform::isSymmetric() const {
  return patTRUE ;
}
