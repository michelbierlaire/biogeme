//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLog.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 14:01:16 2009
//
//--------------------------------------------------------------------

#ifndef bioArithLog_h
#define bioArithLog_h

class bioArithCompositeLiteral ;
#include "bioArithUnaryExpression.h"

/*!
Class implementing a node of the tree representing a natural logarithm operation
*/
class bioArithLog : public bioArithUnaryExpression {

public:
  
  bioArithLog(bioExpressionRepository* rep,
	      patULong par,
	      patULong left,
	      patError*& err) ;
  ~bioArithLog() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual bioExpression* getDerivative(patULong aLiteral, patError*& err) const ;
  virtual bioArithLog* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithLog* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;


};

#endif
