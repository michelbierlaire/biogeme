//-*-c++-*------------------------------------------------------------
//
// File name : patEnvironmentVariable.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr  8 14:37:52 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include "patDisplay.h"
#include "patEnvironmentVariable.h"
#include "patErrMiscError.h"
patEnvironmentVariable::patEnvironmentVariable(patString name) : theName(name) {
  readFromSystem() ;
}

patEnvironmentVariable::~patEnvironmentVariable() {
}

void patEnvironmentVariable::setValue(patString v) {
  theValue = v ;
}

patString patEnvironmentVariable::getValue() const {
  return theValue ;
}

void patEnvironmentVariable::registerToSystem(patError*& err) {
#if HAVE_SETENV
  int error = setenv(theName.c_str(),theValue.c_str(),1) ;
  if (error != 0) {
    stringstream str ;
    str << "Error in setting environment variable " << theName << " to " << theValue ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
#elif HAVE_PUTENV_S
  int error = _putenv_s(theName.c_str(),theValue.c_str()) ;
  if (error != 0) {
    stringstream str ;
    str << "Error in setting environment variable " << theName << " to " << theValue ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
#else
  err = new patErrMiscError("setenv and _putenv_s are not available on this system") ;
  WARNING(err->describe()) ;
  return ;
#endif
}

patBoolean patEnvironmentVariable::readFromSystem() {
#ifdef HAVE_STDLIB_H
  char* theVariable = std::getenv(theName.c_str()) ;
  if (theVariable != NULL) {
    theValue = patString(theVariable) ;
    return patTRUE ;
  }
  else {
    return patFALSE ;
  }
#else
  WARNING("stdlib.h is not available on this system, and getenv cannot be called") ;
  return patFALSE ;
#endif

}

void patEnvironmentVariable::addDescription(patString str) {
  theDescription = str ;
}

patString patEnvironmentVariable::getDescription() const {
  return theDescription ;
}

patString patEnvironmentVariable::getName() const {
  return theName ;
}

ostream& operator<<(ostream& stream, patEnvironmentVariable& ev) {
  stream << ev.getName() << " = " << ev.getValue() ;
  return stream ;
}
