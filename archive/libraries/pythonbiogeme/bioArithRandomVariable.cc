//-*-c++-*------------------------------------------------------------
//
// File name : bioArithRandomVariable.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr 22 07:37:29 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioArithRandomVariable.h"
#include "bioRandomVariable.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patErrMiscError.h"
#include "bioLiteralRepository.h"

bioArithRandomVariable::bioArithRandomVariable(bioExpressionRepository* rep,
					       patULong par, 
					       patULong uniqueId, 
					       patULong rvId) 
  : bioArithLiteral(rep, par,uniqueId), theRandomVariableId(rvId) {

}
  
bioArithRandomVariable* bioArithRandomVariable::getDeepCopy(bioExpressionRepository* rep,
							    patError*& err) const {
  bioArithRandomVariable* newRV = 
    new bioArithRandomVariable(rep,patBadId,theLiteralId,theRandomVariableId) ;
  return newRV ;
}

bioArithRandomVariable* bioArithRandomVariable::getShallowCopy(bioExpressionRepository* rep,
							    patError*& err) const {
  bioArithRandomVariable* newRV = 
    new bioArithRandomVariable(rep,patBadId,theLiteralId,theRandomVariableId) ;
  return newRV ;
}

patReal bioArithRandomVariable::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  patReal result = bioLiteralRepository::the()->getRandomVariableValue(theRandomVariableId, getThreadId(), err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return result ;
}



