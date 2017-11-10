//-*-c++-*------------------------------------------------------------
//
// File name : bioArithConstant.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Wed May  6 16:35:24 2009
//
//--------------------------------------------------------------------

#ifndef bioArithConstant_h
#define bioArithConstant_h

#include "bioArithElementaryExpression.h"
#include "patType.h"

/*!
Class implementing the node for numerical constants in an expression
*/

class bioArithConstant: public bioArithElementaryExpression {

public:
  bioArithConstant(bioExpressionRepository* rep,
		   patULong par, 
		   patReal aValue) ;
  
 public:
  
  virtual patString getOperatorName() const ;
  virtual patReal getValue(patBoolean prepareGradient,  patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithConstant* getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual bioArithConstant* getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual patBoolean isStructurallyZero() const ;
  virtual patBoolean isStructurallyOne() const ;
  virtual patBoolean isConstant() const ;
  virtual patString getExpression(patError*& err) const ;
  virtual patString getExpressionString() const  ;
  virtual patBoolean dependsOf(patULong aLiteralId) const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

 protected:

  patReal theValue ; 

};

#endif
