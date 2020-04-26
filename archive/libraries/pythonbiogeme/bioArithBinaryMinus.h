//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBinaryMinus.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 20:25:55 2009
//
//--------------------------------------------------------------------

#ifndef bioArithBinaryMinus_h
#define bioArithBinaryMinus_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node for a soustraction operation
*/
class bioArithBinaryMinus : public bioArithBinaryExpression {

public:
  
  bioArithBinaryMinus(bioExpressionRepository* rep,
		      patULong par,
		      patULong left, 
		      patULong right,
		      patError*& err) ;
  ~bioArithBinaryMinus() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithBinaryMinus* getDeepCopy(bioExpressionRepository* rep,
					   patError*& err) const ;
  virtual bioArithBinaryMinus* getShallowCopy(bioExpressionRepository* rep,
					   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian, 				  patBoolean debugDerivatives,
patError*& err) ;

};

#endif
