//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNormalCdf.h
// Author :    Michel Bierlaire
// Date :      Tue Jun  8 17:59:01 2010
//
//--------------------------------------------------------------------

#ifndef bioArithNormalCdf_h
#define bioArithNormalCdf_h


#include "bioArithUnaryExpression.h"
#include "patNormalCdf.h"

/*!
Class implementing a node of the tree representing the CDF of the normal distribution
*/
class bioArithNormalCdf : public bioArithUnaryExpression {

public:
  
  bioArithNormalCdf(bioExpressionRepository* rep,
		    patULong par,
		    patULong left,
		    patError*& err) ;
  ~bioArithNormalCdf() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteral, patError*& err) const ;
  virtual bioArithNormalCdf* getDeepCopy(bioExpressionRepository* rep,
					 patError*& err) const ;
  virtual bioArithNormalCdf* getShallowCopy(bioExpressionRepository* rep,
					 patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

private:
  patReal pi ;
  patReal invSqrtTwoPi ;
};

#endif
