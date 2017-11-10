//-*-c++-*------------------------------------------------------------
//
// File name : bioArithDerivative.h
// Author :    Michel Bierlaire
// Date :      Tue May 11 06:06:10 2010
//
//--------------------------------------------------------------------

#ifndef bioArithDerivative_h
#define bioArithDerivative_h


#include "bioArithUnaryExpression.h"
#include "patLap.h"

/*!
Class implementing a node of the tree representing the derivative of the expression with respect to a literal
*/
class bioArithDerivative : public bioArithUnaryExpression {

public:
  
  bioArithDerivative(bioExpressionRepository* rep,
		     patULong par,
		     patULong c,
		     patString aLiteralName,
		     patError*& err) ;
  
  ~bioArithDerivative() ;

public:
  virtual patString getExpression(patError*& err) const ;
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithDerivative* getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual bioArithDerivative* getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
public:
  patULong literalId ;
  patString literalName ;
  bioExpression* theDerivative ;
  patULong theDerivativeId ;

};

#endif
