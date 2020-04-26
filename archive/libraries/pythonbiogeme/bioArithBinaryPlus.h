//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBinaryPlus.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Apr 28 15:11:56 2009
//
//--------------------------------------------------------------------

#ifndef bioArithBinaryPlus_h
#define bioArithBinaryPlus_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node for an addition operation
*/
class bioArithBinaryPlus : public bioArithBinaryExpression {

public:
  
  bioArithBinaryPlus(bioExpressionRepository* rep,
		     patULong par,
		     patULong left, 
		     patULong right,
		     patError*& err) ;
  ~bioArithBinaryPlus() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithBinaryPlus* getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual bioArithBinaryPlus* getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual patString getExpression(patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
 };

#endif
