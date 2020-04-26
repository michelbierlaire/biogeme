//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMin.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 16:50:49 2010
//
//--------------------------------------------------------------------

#ifndef bioArithMin_h
#define bioArithMin_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a max operation
*/
class bioArithMin : public bioArithBinaryExpression {

public:
  
  bioArithMin(bioExpressionRepository* rep,
	      patULong par,
	      patULong left,
	      patULong right,
	      patError*& err) ;
  ~bioArithMin() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithMin* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithMin* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
