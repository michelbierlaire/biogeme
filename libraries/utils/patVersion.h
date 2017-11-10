//-*-c++-*------------------------------------------------------------
//
// File name : patVersion.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Mon Jul  9 15:13:45 2001
//
//--------------------------------------------------------------------

#ifndef patVersion_h
#define patVersion_h

#include "patString.h" 

/**
@doc Provides version information. This is a singleton object, so that the
version number is guaranteed to be consistent throughout the code.
  @author \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}, EPFL (Mon Jul  9 15:13:45 2001)
*/

class patVersion {

  friend class patSingletonFactory ;
public:
  /**
   */
  static patVersion* the() ;
  /**
   */
  patString getVersion() const ;
  /**
   */
  patString getDate() const ;
  /**
   */
  patString getVersionInfoDate() const ;
  /**
   */
  patString getVersionInfo() const ;
  /**
   */
  patString getVersionDate() const ;
  /**
   */
  patString getVersionInfoAuthor() const ;
  /**
   */
  patString getVersionInfoCompiled() const ;
  /**
   */
  patString getVersionInfoCompiledDate() const ;
  /**
   */
  patString getCopyright() const ;

  /**
   */
  patString getLicense() const ;

  /**
   */
  patString getUser() const ;

private:
  patVersion() ;
  patString version ;
  patString date ;
  patString user ;
  patString copyright ;
  patString versionInfo ;
  patString versionDate ;
  patString versionInfoDate ;
  patString versionInfoAuthor ;
  patString versionInfoCompiled ;
};

#endif
