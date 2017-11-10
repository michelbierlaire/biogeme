//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnifRandom.h
// Author :    Michel Bierlaire
// Date :      Wed May 13 16:12:59 2015
//
//--------------------------------------------------------------------

#ifndef bioArithUnifRandom_h
#define bioArithUnifRandom_h

#include "bioRandomDraws.h"
#include "bioArithUnaryExpression.h"

/*!
Class defining an interface for a random parameter uusing draws
*/

// The child computes the number of the individual or the observation. Typically, it is read in the file.
class bioArithUnifRandom : public bioArithUnaryExpression {

public:

  bioArithUnifRandom(bioExpressionRepository* rep,
		     patULong par, 
		     patULong individual, 
		     patULong id, 
		     bioRandomDraws::bioDrawsType  type,
		     patError*& err) ;
  
  ~bioArithUnifRandom() ;
  patString getOperatorName() const  ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  bioArithUnifRandom* getDeepCopy(bioExpressionRepository* rep, 
				  patError*& err) const ;
  bioArithUnifRandom* getShallowCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
  patULong getNumberOfOperations() const ;
  patString getExpressionString() const  ;
  patString getExpression(patError*& err) const  ;
  patBoolean dependsOf(patULong aLiteralId) const  ;
  virtual bioFunctionAndDerivatives* 
  getNumericalFunctionAndGradient(vector<patULong> literalIds, 
				  patBoolean computeHessian, 
				  patBoolean debugDerivatives,
				  patError*& err) ;
  
  
protected:
  patULong variableIdForDraws ;
  bioRandomDraws::bioDrawsType type ;

};


#endif
