//-*-c++-*------------------------------------------------------------
//
// File name : patErrMiscError.cc
// Author :   Michel Bierlaire
// Date :     Mon Dec 21 14:37:15 1998
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
#include "patErrMiscError.h"
#include "patDisplay.h"

patErrMiscError::patErrMiscError(const string& comment) {
  comment_ = comment ;
}

string patErrMiscError::describe() {
  string out = "Error: " ;
  out += comment_ ;
  return out ;
}

patBoolean patErrMiscError::tryToRepair() {
  WARNING("Sorry. I don't know how to repair that error.") ;
  return patFALSE ;
}
