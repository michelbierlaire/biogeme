//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMult.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 20:41:16 2009
//
//--------------------------------------------------------------------

#ifndef bioArithMult_h
#define bioArithMult_h

class bioArithCompositeLiteral ;
#include "bioArithBinaryExpression.h"

/*!
Class implementing a node for a multiplication operation
*/
class bioArithMult : public bioArithBinaryExpression {

public:
  
  bioArithMult(bioExpressionRepository* rep,
	       patULong par,
	       patULong left, 
	       patULong right,
	       patError*& err) ;
  ~bioArithMult() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual bioExpression* getDerivative(patULong aLiteral, 
				       patError*& err) const ;

  virtual bioArithMult* getDeepCopy(bioExpressionRepository* rep,
				    patError*& err) const;
  virtual bioArithMult* getShallowCopy(bioExpressionRepository* rep,
				    patError*& err) const;
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

};

#endif
