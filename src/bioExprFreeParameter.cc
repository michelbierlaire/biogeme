//-*-c++-*------------------------------------------------------------
//
// File name : bioExprFreeParameter.cc
// @date   Thu Jul  4 18:58:06 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprFreeParameter.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprFreeParameter::bioExprFreeParameter(bioUInt literalId, bioUInt parameterId,bioString name) : bioExprLiteral(literalId,name), theParameterId(parameterId) {
  
}
bioExprFreeParameter::~bioExprFreeParameter() {
}

bioString bioExprFreeParameter::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << " lit[" << theLiteralId << "],free[" << theParameterId << "]" ;
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

bioReal bioExprFreeParameter::getLiteralValue() const {
  if (parameters == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"parameters") ;
  }
  if (theParameterId >= parameters->size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,theParameterId,0,parameters->size() - 1) ;
  }
  return (*parameters)[theParameterId] ;
}


