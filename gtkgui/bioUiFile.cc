//-*-c++-*------------------------------------------------------------
//
// File name : bioUiFile.cc
// Author :    Michel Bierlaire
// Date :      Tue Mar 14 09:13:58 2017
//
//--------------------------------------------------------------------

#include "bioUiFile.h"
#include "patDisplay.h"
#include "patFileExists.h"
#include "patErrMiscError.h"

bioUiFile::bioUiFile(patString name):
  fileName(name) {
  listOfDirectories.push_back(".") ;
  listOfDirectories.push_back(__DATADIR) ;
}

patBoolean bioUiFile::fileFound() {
  for (vector<patString>::iterator i = listOfDirectories.begin();
       i != listOfDirectories.end() ;
       ++i) {
    patString ff = *i + "/" + fileName ;
    DEBUG_MESSAGE("Checking " << ff) ;
    if (patFileExists()(ff)) {
      fullPath = ff ;
      return patTRUE ;
    }
  }
  return patFALSE ;
}

patString bioUiFile::getUiFile(patError*& err) {
  if (fileFound()) {
    return fullPath ;
  }
  stringstream str ;
  str << "File " << fileName << " not found in the following directories: " ;
  for (vector<patString>::iterator i = listOfDirectories.begin();
       i != listOfDirectories.end() ;
       ++i) {
    if (i != listOfDirectories.begin()) {
      str << ":" ;
    }
    str << *i ;
  }
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patString() ;
}
  
