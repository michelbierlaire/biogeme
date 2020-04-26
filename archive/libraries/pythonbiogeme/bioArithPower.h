//-*-c++-*------------------------------------------------------------
//
// File name : bioArithPower.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May 5  12:05:41 2009
//
//--------------------------------------------------------------------

#ifndef bioArithPower_h
#define bioArithPower_h

class bioArithCompositeLiteral ;
#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a power operation
*/
class bioArithPower : public bioArithBinaryExpression {

public:
  
  bioArithPower(bioExpressionRepository* rep,
		patULong par,
		patULong left,
                patULong right,
		patError*& err) ;

  ~bioArithPower() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithPower* getDeepCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual bioArithPower* getShallowCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getExpressionString() const ;

  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian, 				  patBoolean debugDerivatives,
 patError*& err) ;

private:  
  vector<patBoolean> leftDependsOnLiteral ;
  vector<patBoolean> rightDependsOnLiteral ;
};

#endif
