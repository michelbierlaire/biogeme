//-*-c++-*------------------------------------------------------------
//
// File name : bioArithExp.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 14:13:31 2009
//
//--------------------------------------------------------------------

#ifndef bioArithExp_h
#define bioArithExp_h

class bioArithCompositeLiteral ;
#include "bioArithUnaryExpression.h"

/*!
Class implementing a node of the tree representing an exponential operation
*/
class bioArithExp : public bioArithUnaryExpression {

public:
  
  bioArithExp(bioExpressionRepository* rep,
	      patULong par,
	      patULong left,
	      patError*& err) ;
  ~bioArithExp() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithExp* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithExp* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

};

#endif
