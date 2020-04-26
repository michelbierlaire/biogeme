//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLess.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:28:36 2009
//
//--------------------------------------------------------------------

#ifndef bioArithLess_h
#define bioArithLess_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (<) operation
*/
class bioArithLess : public bioArithBinaryExpression {

public:
  
  bioArithLess(bioExpressionRepository* rep,
	       patULong par,
	       patULong left,
	       patULong right,
	       patError*& err) ;
  ~bioArithLess() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual bioArithLess* getDeepCopy(bioExpressionRepository* rep,
				    patError*& err) const ;
  virtual bioArithLess* getShallowCopy(bioExpressionRepository* rep,
				    patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
};

#endif
