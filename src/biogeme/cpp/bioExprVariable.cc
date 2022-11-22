//-*-c++-*------------------------------------------------------------
//
// File name : bioExprVariable.cc
// @date   Thu Jul  4 19:25:29 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprVariable.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprVariable::bioExprVariable(bioUInt literalId, bioUInt variableId,bioString name) : bioExprLiteral(literalId,name), theVariableId(variableId) {
  
}
bioExprVariable::~bioExprVariable() {
}

bioString bioExprVariable::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << " lit[" << theLiteralId << "], fixed[" << theVariableId << "]" ;
  if (rowIndex != NULL) {
    str << " <" << *rowIndex << ">" ;
  }
  try {
    getLiteralValue() ;
  }
  catch(bioExceptions& e) {
    return str.str() ;
  }
  str << "(" << getLiteralValue() << ")";
  return str.str() ;
}

bioReal bioExprVariable::getLiteralValue() const {
  bioReal value(missingData) ;
  if (rowIndex == NULL) {
    if (individualIndex == NULL) {
      std::stringstream str ;
      str << "No data has been provided to the formula to obtain a value for variable " << theName ;
      throw bioExceptNullPointer(__FILE__,__LINE__,str.str()) ;
    }
    else {
      // We consider the first observation of this individual
      bioUInt theFirstIndex = (*dataMap)[*individualIndex][0] ;
      if (theVariableId >= (*data)[theFirstIndex].size()) {
	throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,theVariableId,0,(*data)[theFirstIndex].size() - 1) ;
      }
      value = (*data)[theFirstIndex][theVariableId] ;
    }
  }
  else {
    if (*rowIndex >= data->size()) {
      std::stringstream str ;
      str << theName << ": " << *rowIndex << " out of range [0," << data->size() - 1 << "]" ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    if (theVariableId >= (*data)[*rowIndex].size()) {
      std::stringstream str ;
      str << theName
	  << ": "
	  << "Value "
	  << theVariableId
	  << " out of range [0,"
	  << (*data)[*rowIndex].size() - 1
	  <<"]" ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
    value = (*data)[*rowIndex][theVariableId] ;
  }
  if (value == missingData) {
    std::stringstream str ;
    str << "Variable " << theName << " takes value " << missingData << " at row " << *rowIndex << ". This value is interpreted as a missing value by Biogeme. If it is a genuine value, change the parameter 'missingData' in Biogeme. If not, either remove the observation or change the specification of the model." ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  return value ;
}


