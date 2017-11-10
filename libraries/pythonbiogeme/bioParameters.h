//-*-c++-*------------------------------------------------------------
//
// File name : bioParameters.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 17 14:12:02 2009
//
//--------------------------------------------------------------------

#ifndef bioParameters_h
#define bioParameters_h

#include <set>
#include <map>
#include "patError.h"
#include "trParameters.h"
#include "bioParameterIterator.h"

class bioParameters {
  friend class bioPythonSingletonFactory ;
  friend class bioModelParser ;
  friend class bioMain ;
  friend class bioPythonWrapper ;
 public:
  static bioParameters* the() ;
  // Parameters for the trust region algorithm
  trParameters getTrParameters(patError*& err) const ;
  patReal getValueReal(patString p, patError*& err) const ;
  patString getValueString(patString p, patError*& err) const ;
  long getValueInt(patString p, patError*& err) const ;
  patReal getValueReal(patString p) const ;
  patString getValueString(patString p) const ;
  long getValueInt(patString p) const ;
  void readParamFile(patString aFile, patError*& err) ;
  void setParameters(map<patString, patString>* aDict, patError*& err) ;
  patBoolean isRealParam(patString paramName) const ;
  patBoolean isStringParam(patString paramName) const ;
  patBoolean isIntParam(patString paramName) const ;
  patString printPythonCode() const ;
  patString printDocumentation() const ;
  bioParameterIterator<patReal>* createRealIterator() ;
  bioParameterIterator<long int>* createIntegerIterator() ;
  bioParameterIterator<patString>* createStringIterator() ;
 protected:
  bioParameters() ;
  void setValueReal(patString p, patReal v, patError*& err) ;
  void setValueString(patString p, patString v, patError*& err) ;
  void setValueInt(patString p, long v, patError*& err) ;

  patString getDocumentationFilename() const ;
 private:
  // Contains the value as well as the documentation
  map<patString,pair<patReal,patString> > realValues ;
  map<patString,pair<patString,patString> > stringValues ;
  map<patString,pair<long int, patString> > integerValues ;

  patString docFile ;
};

#endif
