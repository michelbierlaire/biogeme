//-*-c++-*------------------------------------------------------------
//
// File name : patVersion.cc
// Author:     Michel Bierlaire, EPFL
// Date  :     Mon Jul  9 15:18:09 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <cassert>
#include "patVersion.h"
#include "patCompilationDate.h"
#include "patSingletonFactory.h"
#include "patUser.h"

patVersion::patVersion() :
  version(PACKAGE_VERSION),
  date(__PATCOMPILATIONDATE) ,
  user(__PATUSER) ,
  copyright("Michel Bierlaire, EPFL") {
  stringstream str1 ;
  str1 << PACKAGE_STRING ;
  versionInfo = patString(str1.str()) ;
  stringstream str4 ;
  str4 << date  ;
  versionDate = patString(str4.str()) ;
  versionInfoDate = patString(str1.str() + " [" + str4.str() + "]") ;
  stringstream str2 ;
  str2 << "Michel Bierlaire, EPFL" ;
  versionInfoAuthor = patString(str2.str()) ;
  stringstream str3 ;
  str3 << "-- Compiled by " << user  ;
  versionInfoCompiled = patString(str3.str()) ;
}
  

patVersion* patVersion::the() {
  return patSingletonFactory::the()->patVersion_the() ;
}

patString patVersion::getVersion() const {
  return version ;
}

patString patVersion::getLicense() const {
  return patString("BIOGEME is distributed free of charge.\nWe ask each user to register to Biogeme's users group,\nand to mention explicitly the use of the package when publishing results,\nusing the following reference:\nBierlaire, M. (2003). BIOGEME: A free package for the estimation of discrete choice models ,\nProceedings of the 3rd Swiss Transportation Research Conference, Ascona, Switzerland.") ;

}


patString patVersion::getDate() const {
  return date ;
}

patString patVersion::getVersionInfoDate() const {
  
  return patString(versionInfoDate) ;
}

patString patVersion::getVersionInfo() const {
  
  return patString(versionInfo) ;
}

patString patVersion::getVersionDate() const {
  
  return patString(versionDate) ;
}

patString patVersion::getVersionInfoAuthor() const {
  return versionInfoAuthor ;
}

patString patVersion::getVersionInfoCompiled() const {
  return versionInfoCompiled ;
}

patString patVersion::getVersionInfoCompiledDate() const {
  
  return patString(versionInfoCompiled+" on "+date) ;
}

patString patVersion::getCopyright() const {
  return copyright ;
}

patString patVersion::getUser() const {
  return user ;
}
