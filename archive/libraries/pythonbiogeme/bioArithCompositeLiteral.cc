//-*-c++-*------------------------------------------------------------
//
// File name : bioArithCompositeLiteral.cc
// Author :    Michel Bierlaire
// Date :      Thu Apr 21 09:54:49 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h" 
#include "bioArithCompositeLiteral.h"
#include "bioCompositeLiteral.h"
#include "bioLiteralRepository.h"

bioArithCompositeLiteral::bioArithCompositeLiteral(bioExpressionRepository* rep,
						   patULong par, 
						   patULong uniqueId, 
						   patULong vId) 
  : bioArithLiteral(rep,par,uniqueId),theCompositeLiteralId(vId) {

}
  
bioArithCompositeLiteral* bioArithCompositeLiteral::getDeepCopy(bioExpressionRepository* rep,
								patError*& err) const {
  bioArithCompositeLiteral* theNewCompositeLiteral = new bioArithCompositeLiteral(rep,patBadId,theLiteralId, theCompositeLiteralId) ;
  return theNewCompositeLiteral ;
  
}

bioArithCompositeLiteral* bioArithCompositeLiteral::getShallowCopy(bioExpressionRepository* rep,
								patError*& err) const {
  bioArithCompositeLiteral* theNewCompositeLiteral = new bioArithCompositeLiteral(rep,patBadId,theLiteralId, theCompositeLiteralId) ;
  return theNewCompositeLiteral ;
  
}

patReal bioArithCompositeLiteral::getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  {
  
  if(currentLap > lastComputedLap || currentLap == patLapForceCompute){
    if (theCompositeLiteralId == patBadId) {
      err = new patErrNullPointer("bioCompositeLiteral") ;
      WARNING(err->describe()) ;
      return patReal() ;
    }
    patReal result = bioLiteralRepository::the()->getCompositeValue(theCompositeLiteralId,getThreadId(),err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    lastValue = result ;
    lastComputedLap = currentLap;
  }

  return lastValue;
}

