//-*-c++-*------------------------------------------------------------
//
// File name : bioCompositeLiteral.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr  5 11:33:08 2011
//
//--------------------------------------------------------------------

#include <sstream>
#include "bioCompositeLiteral.h"
#include "patDisplay.h"
#include "bioParameters.h"

patReal bioCompositeLiteral::getValue(patError*& err) const {
  return currentValue ;
}
  
void bioCompositeLiteral::setValue(patReal val, patError*& err) {
  currentValue = val ;
}

bioCompositeLiteral::bioCompositeLiteral(patString theName,
                                         patULong uniqueId,
					 patULong clId) :
  bioLiteral(theName,uniqueId) , theCompositeLiteralId(clId) {
  patError* err(NULL) ;
  currentValue = bioParameters::the()->getValueReal("missingValue",err) ;
  if (err != NULL) {
    WARNING(err->describe());
  }
}

bioLiteral::bioLiteralType bioCompositeLiteral::getType() const {
  return COMPOSITE ;
}

ostream& operator<<(ostream &str, const bioCompositeLiteral& x) {
  str << "x[" ;
  str << x.theCompositeLiteralId << "]=" << x.name << ";" ;
  str << "ID=" << x.uniqueId ;
  str << " CompId= " << x.theCompositeLiteralId ;
  return str ;
}


patULong bioCompositeLiteral::getCompositeId() const {
  return theCompositeLiteralId ;
}
