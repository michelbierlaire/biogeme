//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMultinaryPlus.h
// Author :    Michel Bierlaire
// Date :      Mon May 23 17:56:41 2011
//
//--------------------------------------------------------------------

#ifndef bioArithMultinaryPlus_h
#define bioArithMultinaryPlus_h


#include "bioArithMultinaryExpression.h"

/*!
Class implementing a node for an addition operation
*/
class bioArithMultinaryPlus : public bioArithMultinaryExpression {

public:
  
  bioArithMultinaryPlus(bioExpressionRepository* rep,
			patULong par,
			vector<patULong> l, 
			patError*& err) ;
  ~bioArithMultinaryPlus() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithMultinaryPlus* getDeepCopy(bioExpressionRepository* rep,
					     patError*& err) const ;
  virtual bioArithMultinaryPlus* getShallowCopy(bioExpressionRepository* rep,
					     patError*& err) const ;
  virtual patString getExpression(patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
 };

#endif
