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

bioExprLiteral::bioExprLiteral(bioUInt literalId, bioString name) : bioExpression(), theLiteralId(literalId), theName(name),first(true),myId(bioBadId),theType(bioExprLiteral::bioUnknown) {
  
}

bioExprLiteral::~bioExprLiteral() {

}

bioDerivatives* bioExprLiteral::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions("If the hessian is needed, the gradient must be computed") ;
  }

  if (theDerivatives == NULL) {
    theDerivatives = new bioDerivatives(literalIds.size()) ;
  }
  else {
    if (gradient && theDerivatives->getSize() != literalIds.size()) {
      delete(theDerivatives) ;
      theDerivatives = new bioDerivatives(literalIds.size()) ;
    }
  }

  theDerivatives->with_g = gradient ;
  theDerivatives->with_h = hessian ;


  theDerivatives->f = getLiteralValue() ;
  if (gradient) {
    for (std::size_t i = 0 ; i < literalIds.size() ; ++i) {
      if (literalIds[i] == theLiteralId) {
	theDerivatives->g[i] = 1.0 ;
      }
      else {
	theDerivatives->g[i] = 0.0 ;
      }
      if (hessian) {
	for (std::size_t j = 0 ; j < literalIds.size() ; ++j) {
	theDerivatives->h[i][j] = 0.0 ;
	}
      }
    }
  }
  return theDerivatives ;
}


bioString bioExprLiteral::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << "[" << theLiteralId << "]" ;
  return str.str() ;
}

bioReal bioExprLiteral::getLiteralValue() {
  if (first) {
    first = false ;
    if (parameters == NULL) {
      throw bioExceptions("Null pointer to vector of parameters") ;
    }
    if (fixedParameters == NULL) {
      throw bioExceptions("Null pointer to vector of fixed parameters") ;
    }
    if (variables == NULL) {
      throw bioExceptions("Null pointer to vector of variables") ;
    }
    if (theLiteralId >= variables->size() + parameters->size() + fixedParameters->size()) {
      throw bioExceptOutOfRange<bioUInt>(theLiteralId,0,variables->size() + parameters->size() + fixedParameters->size() - 1) ;
    }
    myId = theLiteralId ;
    // The numbering is sequential. First the parameters, then the fixed parameters, then the variables
    if (myId >= parameters->size()) {
      myId -= parameters->size() ;
      if (myId >= fixedParameters->size()) {
	myId -= fixedParameters->size() ;
	theType = bioExprLiteral::bioVariable ;
      }
      else {
	theType = bioExprLiteral::bioFixedParameter ;
      }
    }
    else {
      theType = bioExprLiteral::bioParameter ;
    }
  }
  switch(theType) {
  case bioVariable:
    if (myId >= variables->size()) {
      throw bioExceptOutOfRange<bioUInt>(myId,0,variables->size()  - 1) ;
    }
    return (*variables)[myId] ;
  case bioFixedParameter:
    if (myId >= fixedParameters->size()) {
      throw bioExceptOutOfRange<bioUInt>(myId,0,fixedParameters->size()  - 1) ;
    }
    return (*fixedParameters)[myId] ;
  case bioParameter:
    if (myId >= parameters->size()) {
      throw bioExceptOutOfRange<bioUInt>(myId,0,parameters->size() - 1) ;
    }
    return (*parameters)[myId] ;
  default:
    std::stringstream str ;
    str << "Unknown literal type: " << theType ;
    throw bioExceptions(str.str()) ;
  }
}
