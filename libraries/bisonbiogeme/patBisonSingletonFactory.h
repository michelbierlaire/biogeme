//-*-c++-*------------------------------------------------------------
//
// File name : patBisonSingletonFactory.h
// Author :    Michel Bierlaire
// Date :      Sat Jun 11 19:15:04 2016
//
//--------------------------------------------------------------------

#ifndef patBisonSingletonFactory_h
#define patBisonSingletonFactory_h

#include "patModelSpec.h"
#include "patValueVariables.h"

// This object is in charge of managing objects that must have only
// one instance

class patBisonSingletonFactory {
  public:
  virtual ~patBisonSingletonFactory() ;
  void reset() ;

  static patBisonSingletonFactory* the() ;

  patModelSpec* patModelSpec_the() ;
  patValueVariables* patValueVariables_the() ;

 protected:

  patModelSpec* thePatModelSpec ;
  patValueVariables* thePatValueVariables ;
  patBisonSingletonFactory() ;
};

#endif
