//-*-c++-*------------------------------------------------------------
//
// File name : bioPythonSingletonFactory.h
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 18:50:57 2016
//
//--------------------------------------------------------------------

#ifndef bioPythonSingletonFactory_h
#define bioPythonSingletonFactory_h

#include "bioVersion.h"
#include "bioRandomDraws.h"
#include "bioParameters.h"
#include "bioLiteralValues.h"
#include "bioLiteralRepository.h"
#include "bioIteratorInfoRepository.h"

// This object is in charge of managing objects that must have only
// one instance

class bioPythonSingletonFactory {
  public:
  virtual ~bioPythonSingletonFactory() ;
  void reset() ;

  static bioPythonSingletonFactory* the() ;

  bioVersion* bioVersion_the() ;
  bioRandomDraws* bioRandomDraws_the() ;
  bioParameters* bioParameters_the() ;
  bioLiteralValues* bioLiteralValues_the() ;
  bioLiteralRepository* bioLiteralRepository_the() ;
  bioIteratorInfoRepository* bioIteratorInfoRepository_the() ;
 protected:

  bioVersion* theBioVersion ;
  bioRandomDraws* theBioRandomDraws ;
  bioParameters* theBioParameters ;
  bioLiteralValues* theBioLiteralValues ;
  bioLiteralRepository* theBioLiteralRepository ;
  bioIteratorInfoRepository* theBioIteratorInfoRepository ;
  bioPythonSingletonFactory() ;
};

#endif
