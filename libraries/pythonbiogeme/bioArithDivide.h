//-*-c++-*------------------------------------------------------------
//
// File name : bioArithDivide.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 20:57:24 2009
//
//--------------------------------------------------------------------

#ifndef bioArithDivide_h
#define bioArithDivide_h

class bioArithCompositeLiteral ;
#include "bioArithBinaryExpression.h"

/*!
Class implementing a node for a division operation
*/
class bioArithDivide : public bioArithBinaryExpression {

public:
  
  bioArithDivide(bioExpressionRepository* rep,
		 patULong par,
		 patULong left, 
		 patULong right,
		 patError*& err) ;
  ~bioArithDivide() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId,
				       patError*& err) const ;
  virtual bioArithDivide* getDeepCopy(bioExpressionRepository* rep,
				      patError*& err) const ;
  virtual bioArithDivide* getShallowCopy(bioExpressionRepository* rep,
				      patError*& err) const ;
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

private:
  // bioArithCompositeLiteral* theLeftLiteral ;
  // patULong theLeftLiteralCompId ;
  // bioArithCompositeLiteral* theRightLiteral ;
  // patULong theRightLiteralCompId ;
  //  bioArithCompositeLiteral* theRightSquaredLiteral ;
  //  patULong theRightSquaredLiteralCompId ;


};

#endif
