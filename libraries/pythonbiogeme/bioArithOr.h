//-*-c++-*------------------------------------------------------------
//
// File name : bioArithOr.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 10:44:36 2009
//
//--------------------------------------------------------------------

#ifndef bioArithOr_h
#define bioArithOr_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a logical or operation
*/
class bioArithOr : public bioArithBinaryExpression {

public:
  
  bioArithOr(bioExpressionRepository* rep,
	     patULong par,
	     patULong left, 
	     patULong right,
	     patError*& err) ;
  ~bioArithOr() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithOr* getDeepCopy(bioExpressionRepository* rep,
				  patError*& err) const ;
  virtual bioArithOr* getShallowCopy(bioExpressionRepository* rep,
				  patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds,patBoolean computeHessian, 				  patBoolean debugDerivatives,
patError*& err) ;

};

#endif
