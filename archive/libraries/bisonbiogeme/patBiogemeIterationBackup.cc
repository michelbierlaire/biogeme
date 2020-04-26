//-*-c++-*------------------------------------------------------------
//
// File name : patBiogemeIterationBackup.cc
// Author:     Michel Bierlaire, EPFL
// Date  :     Tue Jun 29 17:38:22 2010
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patBiogemeIterationBackup.h"
#include "patModelSpec.h"

void patBiogemeIterationBackup::saveCurrentIteration() {
  patModelSpec::the()->saveBackup() ;
}
