//-*-c++-*------------------------------------------------------------
//
// File name : bioExprRandomVariable.cc
// @date   Wed May  9 17:17:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprRandomVariable.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprRandomVariable::bioExprRandomVariable(bioUInt uniqueId, bioUInt id, bioString name) : bioExprLiteral(uniqueId,name), rvId(id), valuePtr(NULL) {

}
bioExprRandomVariable::~bioExprRandomVariable() {
}

bioString bioExprRandomVariable::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << " lit[" << theLiteralId << "],rv[" << rvId << "]" ;
  return str.str() ;
}

void bioExprRandomVariable::setRandomVariableValuePtr(bioUInt id, bioReal* v) {
  if (rvId == id) {
    valuePtr = v ;
  }
}

bioReal bioExprRandomVariable::getLiteralValue() const {

  if (valuePtr == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"random variable value") ;
  }
  return *valuePtr ;

}
