//-*-c++-*------------------------------------------------------------
//
// File name : bioArithEqual.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:07:32 2009
//
//--------------------------------------------------------------------

#ifndef bioArithEqual_h
#define bioArithEqual_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (==) operation
*/
class bioArithEqual : public bioArithBinaryExpression {

public:
  
  bioArithEqual(bioExpressionRepository* rep,
		patULong par,
		patULong left,
		patULong right,
		patError*& err) ;
  ~bioArithEqual() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithEqual* getDeepCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual bioArithEqual* getShallowCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual patString getExpressionString() const ;

  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
