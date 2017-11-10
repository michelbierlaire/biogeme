//-*-c++-*------------------------------------------------------------
//
// File name : patErrFileNotFound.cc
// Author :   Michel Bierlaire
// Date :     Mon Dec 21 14:37:43 1998
//
// Modification history:
//
// Date                     Author            Description
// ======                   ======            ============
//
//--------------------------------------------------------------------
//

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patErrFileNotFound.h"
#include "patDisplay.h"

patErrFileNotFound::patErrFileNotFound(const string& fileName) :
  patError(), file(fileName) {
}

string patErrFileNotFound::describe() {
  string out = "Pattern: File " ;
  out += file ;
  out += " not found. " ;
  out += comment_ ;
  return out ;
}

patBoolean patErrFileNotFound::tryToRepair() {
  WARNING("Sorry. I don't know how to repair that error.") ;
  return patFALSE ;
}
