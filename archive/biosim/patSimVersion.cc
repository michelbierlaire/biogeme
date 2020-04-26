//-*-c++-*------------------------------------------------------------
//
// File name : patSimVersion.cc
// Author:     Michel Bierlaire, EPFL
// Date  :     Mon Sep  2 19:07:37 2002
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <cassert>
#include "patSimVersion.h"

patSimVersion::patSimVersion() :
  versionMajor(1),
  versionMinor(0),
  date("in development") {
  stringstream str ;
  str << "BioSim Version " << versionMajor << "." << versionMinor 
      << " [" << date << "]" << '\0' ;
  versionInfo = patString(str.str()) ;
}
  

patSimVersion* patSimVersion::the() {
  static patSimVersion* singleInstance = NULL ;
  if (singleInstance == NULL) {
    singleInstance = new patSimVersion() ;
    assert(singleInstance != NULL) ;
  }
  return singleInstance ;
}

short patSimVersion::getVersionMajor() const {
  return versionMajor ;
}

short patSimVersion::getVersionMinor() const {
  return versionMinor ;
}

patString patSimVersion::getDate() const {
  return date ;
}

patString patSimVersion::getVersionInfo() const {
  return versionInfo ;
}

