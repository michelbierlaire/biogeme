//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNotEqual.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:40:41 2009
//
//--------------------------------------------------------------------

#ifndef bioArithNotEqual_h
#define bioArithNotEqual_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (!=) operation
*/
class bioArithNotEqual : public bioArithBinaryExpression {

public:
  
  bioArithNotEqual(bioExpressionRepository* rep, 
		   patULong par,
		   patULong left,
		   patULong right,
		   patError*& err) ;
  ~bioArithNotEqual() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteral, 
				       patError*& err) const ;
  virtual bioArithNotEqual* getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual bioArithNotEqual* getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
