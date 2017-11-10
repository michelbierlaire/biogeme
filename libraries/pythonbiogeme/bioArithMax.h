//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMax.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sat Feb  6 16:47:16 2010
//
//--------------------------------------------------------------------

#ifndef bioArithMax_h
#define bioArithMax_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a max operation
*/
class bioArithMax : public bioArithBinaryExpression {

public:
  
  bioArithMax(bioExpressionRepository* rep, patULong par,
	      patULong left,
	      patULong right,
	      patError*& err) ;
  ~bioArithMax() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithMax* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithMax* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
