//-*-c++-*------------------------------------------------------------
//
// File name : bioArithAbs.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:53:11 2009
//
//--------------------------------------------------------------------

#ifndef bioArithAbs_h
#define bioArithAbs_h


#include "bioArithUnaryExpression.h"

/*!
Class implementing a node of the tree representing an absolute value operation
*/
class bioArithAbs : public bioArithUnaryExpression {

public:
  
  bioArithAbs(bioExpressionRepository* rep, 
	      patULong par,
	      patULong c,
	      patError*& err) ;
  ~bioArithAbs() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithAbs* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithAbs* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
