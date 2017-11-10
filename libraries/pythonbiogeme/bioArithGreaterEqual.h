//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGreaterEqual.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:31:26 2009
//
//--------------------------------------------------------------------

#ifndef bioArithGreaterEqual_h
#define bioArithGreaterEqual_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (>=) operation
*/
class bioArithGreaterEqual : public bioArithBinaryExpression {

public:
  
  bioArithGreaterEqual(bioExpressionRepository* rep,
		       patULong par,
		       patULong left,
		       patULong right,
		       patError*& err) ;
  ~bioArithGreaterEqual() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;

  virtual bioArithGreaterEqual* getDeepCopy(bioExpressionRepository* rep,
					    patError*& err) const;
  virtual bioArithGreaterEqual* getShallowCopy(bioExpressionRepository* rep,
					    patError*& err) const;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
