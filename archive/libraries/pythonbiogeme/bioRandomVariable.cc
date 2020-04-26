//-*-c++-*------------------------------------------------------------
//
// File name : bioRandomVariable.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr  1 12:21:34 2010
//
//--------------------------------------------------------------------

#include <sstream>
#include "patDisplay.h"
#include "bioRandomVariable.h"
#include "patErrMiscError.h"
 
bioRandomVariable::bioRandomVariable(patString theName, patULong uniqueId, patULong rvId) : 
  bioLiteral(theName,uniqueId), theRandomVariableId(rvId) {
}

bioLiteral::bioLiteralType bioRandomVariable::getType() const {
  return RANDOM ;
}

