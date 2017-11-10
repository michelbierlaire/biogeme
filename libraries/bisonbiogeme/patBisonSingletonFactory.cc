//-*-c++-*------------------------------------------------------------
//
// File name : patBisonSingletonFactory.cc
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 18:16:59 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include "patConst.h"

#include "patBisonSingletonFactory.h"

patBisonSingletonFactory::patBisonSingletonFactory() : 
  thePatModelSpec(NULL),
  thePatValueVariables(NULL) {


}

patBisonSingletonFactory::~patBisonSingletonFactory() {
  reset() ;
}

patBisonSingletonFactory* patBisonSingletonFactory::the() {
  static patBisonSingletonFactory* singleInstance = NULL ;
  if (singleInstance == NULL) {
    singleInstance = new patBisonSingletonFactory() ;
    assert(singleInstance != NULL) ;
  }
  return singleInstance ;
}

patModelSpec* patBisonSingletonFactory::patModelSpec_the() {
  if (thePatModelSpec == NULL) {
    thePatModelSpec = new patModelSpec ;
  }
  return thePatModelSpec ;
}

patValueVariables* patBisonSingletonFactory::patValueVariables_the() {
  if (thePatValueVariables == NULL) {
    thePatValueVariables = new patValueVariables() ;
  }
  return thePatValueVariables ;
}

void patBisonSingletonFactory::reset() {
  DEBUG_MESSAGE("About to delete patModelSpec") ;
  DELETE_PTR(thePatModelSpec) ;
  DEBUG_MESSAGE("About to delete patModelSpec: DONE") ;
  DEBUG_MESSAGE("About to delete patValueVariables") ;
  DELETE_PTR(thePatValueVariables) ;
  DEBUG_MESSAGE("About to delete patValueVariables: DONE") ;
}
