//-*-c++-*------------------------------------------------------------
//
// File name : bioVersion.cc
// Author:     Michel Bierlaire, EPFL
// Date  :     Sun Jun 20 17:22:22 2010
//
//--------------------------------------------------------------------

#include <sstream>
#include <cassert>
#include "bioVersion.h"
#include "bioCompilationDate.h"
#include "bioUser.h"
#include "bioPythonSingletonFactory.h"

#include "config.h"

bioVersion::bioVersion() :
  version(PACKAGE_VERSION),
  date(__BIOCOMPILATIONDATE) ,
  copyright("Michel Bierlaire, EPFL 2001-2014"),
  user(__BIOUSER) {
  stringstream str1 ;
  //  str1 << "BIOGEME Python Version " << versionMajor << "." << versionMinor ;
  str1 << PACKAGE_STRING ;
  versionInfo = patString(str1.str()) ;
  stringstream str4 ;
  str4 << date  ;
  versionDate = patString(str4.str()) ;
  stringstream str2 ;
  str2 << "<a href='http://people.epfl.ch/michel.bierlaire'>Michel Bierlaire</a>, <a href='http://transp-or.epfl.ch'>Transport and Mobility Laboratory</a>, <a href='http://www.epfl.ch'>Ecole Polytechnique F&eacute;d&eacute;rale de Lausanne (EPFL)</a>" ;
  versionInfoAuthor = patString(str2.str()) ;
  stringstream str3 ;
  str3 << "-- Compiled by " << user  ;
  versionInfoCompiled = patString(str3.str()) ;
  
}
  

bioVersion* bioVersion::the() {
  return bioPythonSingletonFactory::the()->bioVersion_the() ;
}

patString bioVersion::getVersion() const {
  return version ;
}

patString bioVersion::getDate() const {
  return date ;
}

patString bioVersion::getVersionInfoDate() const {
  
  return patString(versionInfo + " [" + versionDate + "]") ;
}

patString bioVersion::getVersionInfo() const {
  
  return patString(versionInfo) ;
}

patString bioVersion::getVersionDate() const {
  
  return patString(versionDate) ;
}

patString bioVersion::getVersionInfoAuthor() const {
  return versionInfoAuthor ;
}

patString bioVersion::getVersionInfoCompiled() const {
  return versionInfoCompiled ;
}

patString bioVersion::getCopyright() const {
  return copyright ;
}

patString bioVersion::getUser() const {
  return user ;
}
patString bioVersion::getUrl() const {
  return patString(PACKAGE_URL) ;
}

patString bioVersion::getUrlUsersGroup() const {
  return patString(PACKAGE_BUGREPORT) ;
}
