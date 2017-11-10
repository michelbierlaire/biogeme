//-*-c++-*------------------------------------------------------------
//
// File name : bioVersion.h
// Author:     Michel Bierlaire, EPFL
// Date  :     Sun Jun 20 17:21:45 2010
//
//--------------------------------------------------------------------

#ifndef bioVersion_h
#define bioVersion_h

#include "patString.h" 

/**
@doc Provides version information. This is a singleton object, so that the
version number is guaranteed to be consistent throughout the code.
*/

class bioVersion {

  friend class bioPythonSingletonFactory ;

public:
  /**
   */
  static bioVersion* the() ;
  /**
   */
  patString getVersion() const ;
  /**
   */
  patString getUrl() const ;
  /**
   */
  patString getUrlUsersGroup() const ;
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
  patString getCopyright() const ;

  /**
   */
  patString getUser() const ;

private:
  bioVersion() ;
  patString version ;
  patString date ;
  patString versionInfo ;
  patString versionDate ;
  patString versionInfoAuthor ;
  patString versionInfoCompiled ;
  patString copyright ;
  patString user ;
};

#endif
