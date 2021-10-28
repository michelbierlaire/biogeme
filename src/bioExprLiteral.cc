//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLiteral.cc
// @date   Thu Apr 12 11:33:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprLiteral.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprLiteral::bioExprLiteral(bioUInt literalId, bioString name) : bioExpression(), theLiteralId(literalId), theName(name) {
}

bioExprLiteral::~bioExprLiteral() {
}


const bioDerivatives* bioExprLiteral::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;
  
  theDerivatives.resize(literalIds.size()) ;

  
  if (gradient) {
    theDerivatives.setDerivativesToZero() ;
    for (std::size_t i = 0 ; i < literalIds.size() ; ++i) {
      if (literalIds[i] == theLiteralId) {
	theDerivatives.g[i] = 1.0 ;
      }
      else {
	theDerivatives.g[i] = 0.0 ;
      }
    }
  }
  theDerivatives.f = getLiteralValue() ;
  return &theDerivatives ;
}


bioString bioExprLiteral::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << "[" << theLiteralId << "]" ;
  if (rowIndex != NULL) {
    str << " <" << *rowIndex << ">" ;
  }
  try {
    str << "(" << getLiteralValue() << ")";
  }
  catch(bioExceptions& e) {
    
  }
  return str.str() ;
}

bioBoolean bioExprLiteral::containsLiterals(std::vector<bioUInt> literalIds) const {
  for (std::vector<bioUInt>::const_iterator i = literalIds.begin() ;
       i != literalIds.end() ;
       ++i) {
    if (*i == theLiteralId) {
      return true ;
    }
  }
  return false ;
}

void bioExprLiteral::setData(std::vector< std::vector<bioReal> >* d) {
  data = d ;
  if (data == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"data") ;
  }
  if (data->empty()) {
    throw bioExceptions(__FILE__,__LINE__,"No data") ;
  }
}


std::map<bioString,bioReal> bioExprLiteral::getAllLiteralValues() {
  std::map<bioString,bioReal> m ;
  bioReal v ;
  try {
    v = getLiteralValue() ;
  }
  catch(bioExceptions& e) {
    return m ;
  }
  
  m[theName] = v ;
  return m ;
}

bioUInt bioExprLiteral::getLiteralId() const {
  return theLiteralId ;
}
