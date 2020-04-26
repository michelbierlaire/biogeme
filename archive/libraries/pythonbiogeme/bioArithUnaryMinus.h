//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnaryMinus.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri May 15 09:38:34 2009
//
//--------------------------------------------------------------------

#ifndef bioArithUnaryMinus_h
#define bioArithUnaryMinus_h


#include "bioArithUnaryExpression.h"

/*!
Class implementing a node of the tree representing a unary minus operation
*/
class bioArithUnaryMinus : public bioArithUnaryExpression {

public:
  
  bioArithUnaryMinus(bioExpressionRepository* rep,
		     patULong par,
                     patULong left,
		     patError*& err) ;
  ~bioArithUnaryMinus() ;
  
public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  // virtual bioArithListOfExpressions* 
  // getFunctionAndDerivatives(vector<patULong> literalIds, 
  // 			    patError*& err) const ;
  virtual bioArithUnaryMinus* getDeepCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual bioArithUnaryMinus* getShallowCopy(bioExpressionRepository* rep,
					  patError*& err) const ;
  virtual patString getExpressionString() const ;

  virtual bioFunctionAndDerivatives* 
  getNumericalFunctionAndGradient(vector<patULong> literalIds, 
				  patBoolean computeHessian, 
				  patBoolean debugDerivatives,
				  patError*& err) ;

};

#endif
