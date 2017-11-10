//-*-c++-*------------------------------------------------------------
//
// File name : patSingletonFactory.h
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 18:12:40 2016
//
//--------------------------------------------------------------------

#ifndef patSingletonFactory_h
#define patSingletonFactory_h

#include "patFileNames.h"
#include "patOutputFiles.h"
#include "patVersion.h"
#include "patTimer.h"
#include "patNormalCdf.h"

// This object is in charge of managing objects that must have only
// one instance

class patSingletonFactory {
  public:
  virtual ~patSingletonFactory() ;
  void reset() ;

  static patSingletonFactory* the() ;

  patFileNames* patFileNames_the() ;
  patOutputFiles* patOutputFiles_the() ;
  patVersion* patVersion_the() ;
  patTimer* patTimer_the() ;
  patNormalCdf* patNormalCdf_the() ;

 protected:

  patFileNames* thePatFileNames ;
  patOutputFiles* thePatOutputFiles ;
  patVersion* thePatVersion ;
  patTimer* thePatTimer ;
  patNormalCdf* thePatNormalCdf ;
  patSingletonFactory() ;
};

#endif
