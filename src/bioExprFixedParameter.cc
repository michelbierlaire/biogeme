//-*-c++-*------------------------------------------------------------
//
// File name : bioExprFixedParameter.cc
// @date   Thu Jul  4 19:16:33 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include "bioExprFixedParameter.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprFixedParameter::bioExprFixedParameter(bioUInt literalId, bioUInt parameterId,bioString name) : bioExprLiteral(literalId,name), theParameterId(parameterId) {

  
}
bioExprFixedParameter::~bioExprFixedParameter() {
}



bioString bioExprFixedParameter::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << " lit[" << theLiteralId << "],fixed[" << theParameterId << "]" ;
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

bioReal bioExprFixedParameter::getLiteralValue() const {
  if (fixedParameters == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"fixedParameters") ;
  }
  if (theParameterId >= fixedParameters->size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,theParameterId,0,fixedParameters->size()  - 1) ;
  }
  return (*fixedParameters)[theParameterId] ;
}

