//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteralValues.cc
// Author :    Michel Bierlaire
// Date :      Sun Oct 17 11:43:47 2010
//
//--------------------------------------------------------------------

#include "bioLiteralValues.h"
#include "patDisplay.h"
#include "patErrMiscError.h"
#include "bioPythonSingletonFactory.h"

bioLiteralValues::bioLiteralValues()  {
 
}

bioLiteralValues* bioLiteralValues::the() {
  return bioPythonSingletonFactory::the()->bioLiteralValues_the() ;
}


void bioLiteralValues::setValue(patString l, patReal v, patError*& err) {
  values[l] = v ;
}

patReal bioLiteralValues::getValue(patString l, patError*& err) {
  DEBUG_MESSAGE("Get value of " << l) ;
  map<patString,patReal>::iterator found = values.find(l) ;
  if (found == values.end()) {
    stringstream str ;
    str << "Literal " << l << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return found->second ;
}

void bioLiteralValues::eraseValues() {
  values.erase(values.begin(),values.end()) ;
}

