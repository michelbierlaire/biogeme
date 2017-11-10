//-*-c++-*------------------------------------------------------------
//
// File name : bioArithAnd.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:34:04 2009
//
//--------------------------------------------------------------------

#ifndef bioArithAnd_h
#define bioArithAnd_h


#include "bioArithBinaryExpression.h"
/*!
Class implementing a node of the tree representing a logical and operation
*/
class bioArithAnd : public bioArithBinaryExpression {

public:
  
  bioArithAnd(bioExpressionRepository* rep,
	      patULong par,
	      patULong left, 
	      patULong right,
	      patError*& err) ;
  ~bioArithAnd() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithAnd* getDeepCopy(bioExpressionRepository* rep, 
				   patError*& err) const ;
  virtual bioArithAnd* getShallowCopy(bioExpressionRepository* rep, 
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
