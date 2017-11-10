//-*-c++-*------------------------------------------------------------
//
// File name : patEnvPathVariable.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr  8 15:26:05 2016
//
//--------------------------------------------------------------------

#include "patEnvPathVariable.h"

patEnvPathVariable::patEnvPathVariable(patString name, char d): 
  patEnvironmentVariable(name),
  delimiter(d) {
  
}

patEnvPathVariable::~patEnvPathVariable() {

}
  
void patEnvPathVariable::setValue(patString v) {
  listOfDirectories = split(v,delimiter) ;
  generateValue() ;
}

patString patEnvPathVariable::getValue() const {
  return patString(theValue) ;
}

void patEnvPathVariable::addPath(patString p) {
  listOfDirectories.push_back(p) ;
  generateValue() ;
}


patULong patEnvPathVariable::getNbrPaths() const {
  return listOfDirectories.size() ;
}

void patEnvPathVariable::generateValue() {
  stringstream str ;
  for (vector<patString>::const_iterator i = listOfDirectories.begin() ;
       i != listOfDirectories.end() ;
       ++i) {
    if (i != listOfDirectories.begin()) {
      str << delimiter ;
    }
    str << *i ;
  }
  theValue = patString(str.str()) ;

}
