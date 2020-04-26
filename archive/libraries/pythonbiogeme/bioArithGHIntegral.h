//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGHIntegral.h
// Author :    Michel Bierlaire
// Date :      Tue Mar 30 17:33:52 2010
//
//--------------------------------------------------------------------

#ifndef bioArithGHIntegral_h
#define bioArithGHIntegral_h


#include "bioArithUnaryExpression.h"

/*!
Class implementing a node of the tree representing an integral between -infty and +infty
*/
class bioArithGHIntegral : public bioArithUnaryExpression {

public:
  
  bioArithGHIntegral(bioExpressionRepository* rep,
		   patULong par,
		   patULong left,
		   patString aLiteralName,
		   patError*& err) ;
  
  ~bioArithGHIntegral() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithGHIntegral* getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual bioArithGHIntegral* getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
public:
  patULong literalId ;
  patString literalName ;

private:
  vector<patULong> theCompositeLiteralsIds ;

};

#endif
