//-*-c++-*------------------------------------------------------------
//
// File name : patNormal.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Thu Mar  6 16:56:31 2003
//
//--------------------------------------------------------------------

#include "patMath.h"
#include "patNormal.h"
#include "patUniform.h" 
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "patFileNames.h"


patNormal::patNormal(patBoolean dumpDrawsOnFile) : 
  patRandomNumberGenerator(dumpDrawsOnFile),
  first(patTRUE), 
  uniformNumberGenerator(NULL),
  logFile(NULL) {
  patError* err(NULL) ;
  if (dumpDrawsOnFile) {
    logFile = new ofstream(patFileNames::the()->getNormalDrawLogFile(err).c_str()) ;
  }
}

patNormal::~patNormal() {
  if (logFile != NULL) {
    logFile->close() ;
    DELETE_PTR(logFile) ;
  }
}


void patNormal::setUniform(patUniform* rng) {
  uniformNumberGenerator = rng ;
}

pair<patReal,patReal> patNormal::getNextValue(patError*& err ) {
  if (uniformNumberGenerator == NULL) {
    err = new patErrMiscError("No pseudo-random generator specified") ;
    WARNING(err->describe());
    return pair<patReal,patReal> () ;
  }

  pair<patReal,patReal> result ;
  patReal rn1(0.0) ;
  patReal rn2(0.0) ;
  if (first) {
    rn1 = uniformNumberGenerator->getUniform(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return pair<patReal,patReal>() ;
    }
    rn2 = uniformNumberGenerator->getUniform(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return pair<patReal,patReal>() ;
    }
    patReal r  = sqrt(-2.0 * log(rn1));
    patReal u  = 2.0 * M_PI * rn2;
    v1 = r * cos(u) ;
    v2 = r * sin(u) ;
    first = patFALSE ;    
    result.first =  v1 ;
    result.second = rn1 ;
  }
  else {
    first = patTRUE ;
    result.first = v2 ;
    result.second = rn2;
  }
  if (logFile != NULL) {
    *logFile << result.first << '\t' << result.second << endl ;
  }
  return result ;
}

patBoolean patNormal::isSymmetric() const {
  return patTRUE ;
}

patBoolean patNormal::isNormal() const {
  return patTRUE ;
}
