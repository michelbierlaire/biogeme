//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMonteCarlo.h
// Author :    Michel Bierlaire 
// Date :      Sun May 10 09:16:02 2015
//
//--------------------------------------------------------------------

#ifndef bioArithMonteCarlo_h
#define bioArithMonteCarlo_h

#include "patKalman.h"

#include "bioArithIterator.h"
#include "bioLiteral.h"
#include "bioIteratorInfo.h"


/*!
Class implementing a node of the tree representing an integration expression, using Monte Carlo simulation. It is basically a sum over all draws, divided by the number of draws. 
*/
class bioArithMonteCarlo : public bioArithUnaryExpression {

public:
  
  bioArithMonteCarlo(bioExpressionRepository* rep,
		     patULong par,
		     patULong left,
		     patError*& err) ;

  // Ctor when control variate is applied
  bioArithMonteCarlo(bioExpressionRepository* rep,
		     patULong par,
		     patULong left,
		     patULong theIntegrand,
		     patULong theIntegral,
		     patError*& err) ;
  
  ~bioArithMonteCarlo() ;

  patBoolean containsMonteCarlo() const ;

  void checkMonteCarlo(patBoolean insideMonteCarlo, patError*& err) ;

public:
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;

  virtual patString getOperatorName() const ;
  virtual bioArithMonteCarlo* getDeepCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual bioArithMonteCarlo* getShallowCopy(bioExpressionRepository* rep,
				   patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)   ;
  virtual patULong getNumberOfOperations() const ;

  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

  void calculateControlVariate(patError*& err) ;

  bioFunctionAndDerivatives* getNumericalFunctionAndGradientControlVariate(vector<patULong> literalIds, patBoolean computeHessian,patError*& err) ;

private: 
  bioDrawIterator* theIter ;
  bioExpression* integrand ;
  bioExpression* integral ;
  patBoolean reportControlVariate ;
  patULong numberOfDraws ;
  patKalman* theFilter ;
  vector<patReal> newDrawsDep ;
  vector<patReal> newCvDrawsIndep ;
  patBoolean controlVariate ;

};

#endif
