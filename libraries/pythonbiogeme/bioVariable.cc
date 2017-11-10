//-*-c++-*------------------------------------------------------------
//
// File name : bioVariable.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 10:40:40 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patErrMiscError.h"
#include "patDisplay.h"
#include "bioVariable.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h"
 
bioVariable::bioVariable(patString theName, 
			 patULong uniqueId, 
			 patULong vId,
			 patULong aColId) : 
  bioLiteral(theName,uniqueId),
  theVariableId(vId),
  columnId(aColId) {

}


bioLiteral::bioLiteralType bioVariable::getType() const {
  return VARIABLE ;
}


patULong bioVariable::getColumnId() const {
  return columnId ;
}
