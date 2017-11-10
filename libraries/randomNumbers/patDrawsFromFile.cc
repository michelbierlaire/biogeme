//-*-c++-*------------------------------------------------------------
//
// File name : patDrawsFromFile.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Aug 24 16:55:04 2003
//
//--------------------------------------------------------------------

#include <sstream>
#include "patDisplay.h"
#include "patDrawsFromFile.h"
#include "patErrMiscError.h"

patDrawsFromFile::patDrawsFromFile(const patString& f, 
					       patError*& err) :
  patRandomNumberGenerator(patFALSE),
  theFile(f.c_str()) {
  if (!theFile) {
    stringstream str ;
    str << "Error in opening " << f << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
}


pair<patReal,patReal> patDrawsFromFile::getNextValue(patError*& err) {
  if (theFile.eof()) {
    stringstream str ;
    str << "End of file detected in " << fileName << '\0' ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return pair<patReal,patReal>() ;
  }
  patReal next,nextU ;
  theFile >> next ;
  theFile >> nextU ;
  return pair<patReal,patReal>(next,nextU) ;
}

patBoolean patDrawsFromFile::isSymmetric() const {
  return patFALSE ;
}
