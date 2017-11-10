//-*-c++-*------------------------------------------------------------
//
// File name : bioArithFixedParameter.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr 22 08:15:17 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioArithFixedParameter.h"
#include "bioFixedParameter.h"
#include "patErrNullPointer.h"
#include "bioLiteralRepository.h"

bioArithFixedParameter::bioArithFixedParameter(bioExpressionRepository* rep,
					       patULong par, 
					       patULong uniqueId, 
					       patULong id) 
  : bioArithLiteral(rep,par,uniqueId), theParameterId(id), theParameter(NULL) {

}
  
bioArithFixedParameter* bioArithFixedParameter::getDeepCopy(bioExpressionRepository* rep,
							    patError*& err) const {
  bioArithFixedParameter* theNewParam = new bioArithFixedParameter(rep,patBadId,theLiteralId,theParameterId) ;
  return theNewParam ;
}

bioArithFixedParameter* bioArithFixedParameter::getShallowCopy(bioExpressionRepository* rep,
							    patError*& err) const {
  bioArithFixedParameter* theNewParam = new bioArithFixedParameter(rep,patBadId,theLiteralId,theParameterId) ;
  return theNewParam ;
}

patReal bioArithFixedParameter::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) {
  if (theParameter == NULL) {
    theParameter = bioLiteralRepository::the()->theParameter(theParameterId) ;
  }
  return theParameter->getValue() ;
}


