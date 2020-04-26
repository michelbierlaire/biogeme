//-*-c++-*------------------------------------------------------------
//
// File name : bioArithNormalPdf.h
// Author :    Michel Bierlaire
// Date :      Mon Jun 27 18:19:32 2011
//
//--------------------------------------------------------------------

#ifndef bioArithNormalPdf_h
#define bioArithNormalPdf_h


#include "bioArithUnaryExpression.h"

/*!
We have a dictionary of utilities, mapping values to utilities.
*/
class bioArithNormalPdf : public bioArithUnaryExpression {

public:
  
  bioArithNormalPdf(bioExpressionRepository* rep,
		    patULong par,
		    patULong left,
		    patError*& err) ;

  ~bioArithNormalPdf() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithNormalPdf* getDeepCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual bioArithNormalPdf* getShallowCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual patString getExpressionString() const ;

  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
 patError*& err) ;

private:
  patReal invSqrtTwoPi ;



};

#endif
