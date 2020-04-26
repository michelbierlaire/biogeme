//-*-c++-*------------------------------------------------------------
//
// File name : patSingletonFactory.cc
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 18:16:59 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include "patDisplay.h"
#include "patSingletonFactory.h"


patSingletonFactory::patSingletonFactory() : 
  thePatFileNames(NULL),
  thePatOutputFiles(NULL),
  thePatVersion(NULL),
  thePatTimer(NULL),
  thePatNormalCdf(NULL) {

}

patSingletonFactory::~patSingletonFactory() {
  reset() ;
}

patSingletonFactory* patSingletonFactory::the() {
  static patSingletonFactory* singleInstance = NULL ;
  if (singleInstance == NULL) {
    singleInstance = new patSingletonFactory() ;
    assert(singleInstance != NULL) ;
  }
  return singleInstance ;
}

patFileNames* patSingletonFactory::patFileNames_the() {
  if (thePatFileNames == NULL) {
    thePatFileNames = new patFileNames() ;
  }
  return thePatFileNames ;
}

patOutputFiles* patSingletonFactory::patOutputFiles_the() {
  if (thePatOutputFiles == NULL) {
    thePatOutputFiles = new patOutputFiles() ;
  }
  return thePatOutputFiles ;
}



patVersion* patSingletonFactory::patVersion_the() {
  if (thePatVersion == NULL) {
    thePatVersion = new patVersion() ;
  }
  return thePatVersion ;
}

patTimer* patSingletonFactory::patTimer_the() {
  if (thePatTimer == NULL) {
    thePatTimer = new patTimer() ;
  }
  return thePatTimer ;
}

patNormalCdf* patSingletonFactory::patNormalCdf_the() {
  if (thePatNormalCdf == NULL) {
    thePatNormalCdf = new patNormalCdf() ;
  }
  return thePatNormalCdf ;
}


void patSingletonFactory::reset() {
  DELETE_PTR(thePatFileNames) ;
  DELETE_PTR(thePatOutputFiles) ;
  DELETE_PTR(thePatVersion) ;
  DELETE_PTR(thePatTimer) ;
  DELETE_PTR(thePatNormalCdf) ;
}
