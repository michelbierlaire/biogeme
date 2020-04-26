//-*-c++-*------------------------------------------------------------
//
// File name : patPythonResults.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sat Jul 22 19:08:10 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include <sstream>
#include "patPythonResults.h"

const char* patPythonResults::getTimeStamp() {
  return timeStamp.c_str() ;
}

const char* patPythonResults::getVersion() {
  return version.c_str() ;
}

const char* patPythonResults::getDescription() {
  return description.c_str() ;
}

const char* patPythonResults::getModel() {
  return model.c_str() ;
}

const char* patPythonResults::getDrawsType() {
  return drawsType.c_str() ;
}

const char* patPythonResults::getVarianceCovariance() {
  return varianceCovariance.c_str() ;
}

const char* patPythonResults::getParamName(unsigned int index) {
  if (index >= paramNames.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << paramNames.size()-1 << "]" ;
    WARNING(str.str()) ;
    return("__error") ;
  }
  return paramNames[index].c_str() ;
}

patReal patPythonResults::getEstimate(unsigned int index) {
  if (index >= estimates.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << estimates.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal();
  }
  return estimates[index] ;
}

patReal patPythonResults::getEstimate(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found = 
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknow parameter " << theName) ;
    return patReal() ;
  }
  return estimates[found->second] ;
}

patReal patPythonResults::getStdErr(unsigned int index) {
  if (index >= stdErr.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << stdErr.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return stdErr[index] ;
}

patReal patPythonResults::getStdErr(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return stdErr[found->second] ;
}

patReal patPythonResults::getTTest(unsigned int index) {
  if (index >= tTest.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << tTest.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return tTest[index] ;
}

patReal patPythonResults::getTTest(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return tTest[found->second] ;
}

patReal patPythonResults::getPValue(unsigned int index) {
  if (index >= pValue.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << pValue.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return pValue[index] ;
}

patReal patPythonResults::getPValue(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return pValue[found->second] ;
}

patReal patPythonResults::getStdErrRobust(unsigned int index) {
  if (index >= stdErrRobust.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << stdErrRobust.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return stdErrRobust[index] ;
}

patReal patPythonResults::getStdErrRobust(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return stdErrRobust[found->second] ;
}

patReal patPythonResults::getTTestRobust(unsigned int index) {
  if (index >= tTestRobust.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << tTestRobust.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return tTestRobust[index] ;
}

patReal patPythonResults::getTTestRobust(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return tTestRobust[found->second] ;
}

patReal patPythonResults::getPValueRobust(unsigned int index) {
  if (index >= pValueRobust.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << pValueRobust.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return pValueRobust[index] ;
}

patReal patPythonResults::getPValueRobust(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return pValueRobust[found->second] ;
}

int patPythonResults::  getFixed(unsigned int index) {
  if (index >= fixed.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << fixed.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return fixed[index] ;
}

int patPythonResults::  getFixed(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return fixed[found->second] ;
}

int patPythonResults::  getDistributed(unsigned int index) {
  if (index >= distributed.size()) {
    stringstream str ;
    str << "Index " << index << " out of range [0-" << distributed.size()-1 << "]" ;
    WARNING(str.str()) ;
    return patReal() ;
  }
  return distributed[index] ;
}

int patPythonResults::  getDistributed(const char* name) {
  patString theName(name) ;
  map<patString, unsigned int>::const_iterator found =
    paramIndex.find(theName) ;
  if (found == paramIndex.end()) {
    WARNING("Unknown parameter " << theName) ;
    return patReal() ;
  }
  return distributed[found->second] ;
}

