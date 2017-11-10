//-*-c++-*------------------------------------------------------------
//
// File name : bioLiteral.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 17:23:57 2009
//
//--------------------------------------------------------------------

#include <sstream>
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"
#include "bioArithLiteral.h"
#include "bioArithConstant.h"
#include "bioLiteralRepository.h"
#include "bioLiteralValues.h"
#include "bioExpressionRepository.h"

bioArithLiteral::bioArithLiteral(bioExpressionRepository* rep,
				 patULong par, patULong lId)
  : bioArithElementaryExpression(rep,par), theLiteralId(lId) {
}

patString bioArithLiteral::getOperatorName() const {
  patError* err(NULL) ;
  const bioLiteral* theLiteral = bioLiteralRepository::the()->getLiteral(theLiteralId,err) ;

  if (theLiteral != NULL) {
    return theLiteral->getName() ;
  }
  else {
    return patString("Undefined literal") ;
  }
}


bioExpression* bioArithLiteral::getDerivative(patULong aLiteralId, patError*& err) const {
  const bioLiteral* theLiteral = bioLiteralRepository::the()->getLiteral(theLiteralId,err) ;
  if (theLiteral == NULL) {
    err = new patErrNullPointer("bioLiteral") ;
    WARNING(err->describe()) ;
    return NULL  ;

  }

  if (aLiteralId == theLiteral->getId()) {
    bioExpression* theExpr = theRepository->getOne() ;
    return theExpr ;
  }
  else {
    bioExpression* theExpr = theRepository->getZero() ;
    return theExpr ;
  }
}


patBoolean bioArithLiteral::dependsOf(patULong aLiteralId) const {
  patError* err(NULL) ;
  const bioLiteral* theLiteral = bioLiteralRepository::the()->getLiteral(theLiteralId,err) ;
  if (aLiteralId == theLiteral->getId()) {
    return patTRUE ;
  }
  else {
    return patFALSE ;
  }
}

patString bioArithLiteral::getExpressionString() const {
  patError* err(NULL) ;
  const bioLiteral* theLiteral = bioLiteralRepository::the()->getLiteral(theLiteralId,err) ;
  stringstream str ;
  str << theLiteral->getId() ;
  return patString(str.str()) ;

}

patBoolean bioArithLiteral::isLiteral() const {
  return patTRUE ;
}

patString bioArithLiteral::getExpression(patError*& err) const {
  stringstream str ;
  return getOperatorName() ;
}



bioFunctionAndDerivatives* bioArithLiteral::getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian,patBoolean debugDeriv,  patError*& err) {
  if (result.empty()) {
    result.resize(literalIds.size()) ;
    for (patULong i = 0 ; i < literalIds.size() ; ++i) {
      if (literalIds[i] == theLiteralId) {
	result.theGradient[i] = 1.0 ;
      }
      else {
	result.theGradient[i] = 0.0 ;
      }
    }
  }
  result.theFunction = getValue(patFALSE, patLapForceCompute, err) ;
  return &result ;
}

patBoolean bioArithLiteral::verifyDerivatives(vector<patULong> literalIds, patError*& err)  {
  return patTRUE ;
}
