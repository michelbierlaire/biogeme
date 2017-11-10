//-*-c++-*------------------------------------------------------------
//
// File name : patSimVersion.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Mon Sep  2 19:06:56 2002
//
//--------------------------------------------------------------------

#ifndef patSimVersion_h
#define patSimVersion_h

#include "patString.h" 

/**
@doc Provides version information. This is a singleton object, so that the
version number is guaranteed to be consistent throughout the code.
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Sep  2 19:06:56 2002)
*/

class patSimVersion {

public:
  /**
   */
  static patSimVersion* the() ;
  /**
   */
  short getVersionMajor() const ;
  /**
   */
  short getVersionMinor() const ;
  /**
   */
  patString getDate() const ;
  /**
   */
  patString getVersionInfo() const ;
private:
  patSimVersion() ;
  short versionMajor ;
  short versionMinor ;
  patString date ;
  patString versionInfo ;
};

#endif
