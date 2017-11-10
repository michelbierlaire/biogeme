//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLiteral.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Apr 27 17:19:08 2009
//
//--------------------------------------------------------------------

#ifndef bioArithLiteral_h
#define bioArithLiteral_h

#include "bioArithElementaryExpression.h"
#include "bioLiteral.h"

/*!
Class implementing the node for literals in an expression
*/

class bioArithLiteral: public bioArithElementaryExpression {

public:
  bioArithLiteral(bioExpressionRepository* rep,
		  patULong par, 
		  patULong theLiteralId) ;
  
 public:
  
  virtual patString getOperatorName() const ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual patBoolean dependsOf(patULong aLiteralId) const ;
  virtual patString getExpression(patError*& err) const  ;
  virtual patString getExpressionString() const  ;
  virtual patBoolean isLiteral() const ;

  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

  virtual patBoolean verifyDerivatives(vector<patULong> literalIds, patError*& err)  ;

 protected:

  patULong theLiteralId ;
};

#endif
