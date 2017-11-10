//-*-c++-*------------------------------------------------------------
//
// File name : bioArithVariable.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr 21 07:25:42 2011
//
//--------------------------------------------------------------------

#include "bioArithVariable.h"
#include "bioParameters.h"
#include "bioVariable.h" 
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "bioLiteralRepository.h"

bioArithVariable::bioArithVariable(bioExpressionRepository* rep,
				   patULong par,
				   patULong uniqueId, 
				   patULong vId) :
  bioArithLiteral(rep, par,uniqueId), theVariableId(vId),theVariable(NULL) {

  patError* err(NULL) ;
  checkForMissingValues = (bioParameters::the()->getValueInt("detectMissingValue",err) != 0) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }
  if (checkForMissingValues) {
    missingValue = bioParameters::the()->getValueReal("missingValue",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
    }
  }
}
  
bioArithVariable* bioArithVariable::getDeepCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithVariable* theNewVariable = new bioArithVariable(rep,patBadId,theLiteralId,theVariableId) ;
  return theNewVariable ;
}


bioArithVariable* bioArithVariable::getShallowCopy(bioExpressionRepository* rep,
						patError*& err) const {
  bioArithVariable* theNewVariable = new bioArithVariable(rep,patBadId,theLiteralId,theVariableId) ;
  return theNewVariable ;
}

patReal bioArithVariable::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){

    if (__x == NULL) {
      stringstream str ; 
      str << "Impossible to compute the value of " << *this << " probably because it is not used within an iterator on the data" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    if (theVariable == NULL) {
      theVariable = bioLiteralRepository::the()->theVariable(theVariableId) ;
    }

    if (theVariable == NULL) {
      err = new patErrNullPointer("bioVariable") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }

    if (theVariable->columnId >= __x->size()) {
      err = new patErrOutOfRange<patULong>(theVariable->columnId,0,__x->size()-1) ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    lastValue =  (*__x)[theVariable->columnId] ;
    lastComputedLap = currentLap;
    if (checkForMissingValues) {
      if (lastValue == missingValue) {
	stringstream str ;
	str << "Variable " << theVariable->getName() << " is equal to " << missingValue << ", that is considered to be the code for a missing value" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
    }
    return lastValue;
  }else{
    return lastValue;
  }
}

