//-*-c++-*------------------------------------------------------------
//
// File name : bioPythonSingletonFactory.cc
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 18:16:59 2016
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include "patConst.h"

#include "bioPythonSingletonFactory.h"

bioPythonSingletonFactory::bioPythonSingletonFactory() : 
  theBioVersion(NULL),
  theBioRandomDraws(NULL),
  theBioParameters(NULL),
  theBioLiteralValues(NULL),
  theBioLiteralRepository(NULL),
  theBioIteratorInfoRepository(NULL) {
}

bioPythonSingletonFactory::~bioPythonSingletonFactory() {
  reset() ;
}

bioPythonSingletonFactory* bioPythonSingletonFactory::the() {
  static bioPythonSingletonFactory* singleInstance = NULL ;
  if (singleInstance == NULL) {
    singleInstance = new bioPythonSingletonFactory() ;
    assert(singleInstance != NULL) ;
  }
  return singleInstance ;
}

bioVersion* bioPythonSingletonFactory::bioVersion_the() {
  if (theBioVersion == NULL) {
    theBioVersion = new bioVersion() ;
  }
  return theBioVersion ;
}

bioRandomDraws* bioPythonSingletonFactory::bioRandomDraws_the() {
  if (theBioRandomDraws == NULL) {
    theBioRandomDraws = new bioRandomDraws() ;
  }
  return theBioRandomDraws ;
}

bioParameters* bioPythonSingletonFactory::bioParameters_the() {
  if (theBioParameters == NULL) {
    theBioParameters = new bioParameters() ;
  }
  return theBioParameters ;
}

bioLiteralValues* bioPythonSingletonFactory::bioLiteralValues_the() {
  if (theBioLiteralValues == NULL) {
    theBioLiteralValues = new bioLiteralValues() ;
  }
  return theBioLiteralValues ;
}

bioLiteralRepository* bioPythonSingletonFactory::bioLiteralRepository_the() {
  if (theBioLiteralRepository == NULL) {
    theBioLiteralRepository = new bioLiteralRepository() ;
  }
  return theBioLiteralRepository ;
}

bioIteratorInfoRepository* bioPythonSingletonFactory::bioIteratorInfoRepository_the() {
  if (theBioIteratorInfoRepository == NULL) {
    theBioIteratorInfoRepository = new bioIteratorInfoRepository() ;
  }
  return theBioIteratorInfoRepository ;
}


void bioPythonSingletonFactory::reset() {
  DELETE_PTR(theBioVersion) ;
  DELETE_PTR(theBioRandomDraws) ;
  DELETE_PTR(theBioParameters) ;
  DELETE_PTR(theBioLiteralValues) ;
  DELETE_PTR(theBioLiteralRepository) ;
  DELETE_PTR(theBioIteratorInfoRepository) ;
}
