//-*-c++-*------------------------------------------------------------
//
// File name : patEnvironmentVariable.h
// Author :    Michel Bierlaire
// Date :      Fri Apr  8 14:31:25 2016
//
//--------------------------------------------------------------------

#ifndef patEnvironmentVariable_h
#define patEnvironmentVariable_h
#include "patString.h"
#include "patError.h"
class patEnvironmentVariable {
 public:
  friend ostream& operator<<(ostream& stream, patEnvironmentVariable& ev) ;
  patEnvironmentVariable(patString name) ;
  virtual ~patEnvironmentVariable() ;
  patString getName() const ;
  virtual void setValue(patString v) ;
  virtual patString getValue() const ;
  virtual void registerToSystem(patError*& err) ;
  virtual patBoolean readFromSystem() ;
  virtual void addDescription(patString str) ;
  virtual patString getDescription() const ;
 protected:
  patString theName ;
  patString theDescription ;
  patString theValue ;

};
#endif
