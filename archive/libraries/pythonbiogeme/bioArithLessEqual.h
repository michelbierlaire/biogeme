//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLessEqual.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue May  5 11:34:15 2009
//
//--------------------------------------------------------------------

#ifndef bioArithLessEqual_h
#define bioArithLessEqual_h


#include "bioArithBinaryExpression.h"

/*!
Class implementing a node of the tree representing a comparison (<=) operation
*/
class bioArithLessEqual : public bioArithBinaryExpression {

public:
  
  bioArithLessEqual(bioExpressionRepository* rep,
		    patULong par,
		    patULong left,
		    patULong right, 
		    patError*& err) ;
  ~bioArithLessEqual() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId,
				       patError*& err) const ;
  virtual bioArithLessEqual* getDeepCopy(bioExpressionRepository* rep,
					 patError*& err) const ;
  virtual bioArithLessEqual* getShallowCopy(bioExpressionRepository* rep,
					 patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

};

#endif
