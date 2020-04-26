//-*-c++-*------------------------------------------------------------
//
// File name : bioArithGreater.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:24:53 WEST 2009
//
//--------------------------------------------------------------------

#ifndef bioArithGreater_h
#define bioArithGreater_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (>) operation
*/
class bioArithGreater : public bioArithBinaryExpression {

public:
  
  bioArithGreater(bioExpressionRepository* rep,
		  patULong par,
		  patULong left,
		  patULong right,
		  patError*& err) ;
  ~bioArithGreater() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;

  virtual bioArithGreater* getDeepCopy(bioExpressionRepository* rep,
				       patError*& err) const ;
  virtual bioArithGreater* getShallowCopy(bioExpressionRepository* rep,
				       patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
