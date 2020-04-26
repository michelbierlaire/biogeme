//-*-c++-*------------------------------------------------------------
//
// File name : patFileExists.cc
// Author :    Michel Bierlaire
// Date :      Mon Dec 21 16:27:31 1998
//
//--------------------------------------------------------------------
//

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STAT
#include <sys/stat.h>
#endif

#include <fstream>
#include <iostream>
#include "patDisplay.h"
#include "patFileExists.h"

/// Checks if a file exists using the stat() function
patBoolean patFileExists::operator()(const patString& fileName) {
#ifdef HAVE_STAT
  struct stat buf ;
  int staterr(0) ;
  try {
    staterr = stat(fileName.c_str(),&buf) ;
    return (staterr == 0);
  }
  catch (...) {
    return patFALSE ;
  }
#else
  try {
    ifstream f(fileName.c_str()) ;
    return f.good() ;
  }
  catch (...) {
    return patFALSE ;
  }
#endif
}
