//-*-c++-*------------------------------------------------------------
//
// File name : patErrNullPointer.cc
// Author :   Michel Bierlaire
// Date :     Mon Dec 21 14:36:06 1998
//
//--------------------------------------------------------------------
//
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patErrNullPointer.h"
#include "patDisplay.h"

patErrNullPointer::patErrNullPointer(const string& t): 
  patError(),
  type(t) {

}

string patErrNullPointer::describe() {
  string out = "Error: Pointer to " ;
  out += type  ;
  out += " is NULL. " ;
  out += comment_ ;
  return(out);
}


patBoolean patErrNullPointer::tryToRepair() {
  WARNING("Sorry. I don't know how to repair that error.") ;
  return patFALSE ;
}
