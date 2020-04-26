//-*-c++-*------------------------------------------------------------
//
// File name : patEnvPathVariable.h
// Author :    Michel Bierlaire
// Date :      Fri Apr  8 15:21:24 2016
//
//--------------------------------------------------------------------

#ifndef patEnvPathVariable_h
#define patEnvPathVariable_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "patEnvironmentVariable.h"

class patEnvPathVariable : public patEnvironmentVariable {
 public:
#ifdef IS_WINDOWS
  patEnvPathVariable(patString name, char d=';') ;
#else
  patEnvPathVariable(patString name, char d=':') ;
#endif
  ~patEnvPathVariable() ;
  virtual void setValue(patString v) ;
  virtual patString getValue() const ;
  void addPath(patString p) ;
  patULong getNbrPaths() const ;
 private:
  void generateValue() ;
 private:
  vector<patString> listOfDirectories ;
  char delimiter ;
};
#endif
