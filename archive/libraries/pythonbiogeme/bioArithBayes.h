//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayes.h
// Author :    Michel Bierlaire
// Date :      Tue Jul 31 17:26:09 2012
//
//--------------------------------------------------------------------

#ifndef bioArithBayes_h
#define bioArithBayes_h

#include "patError.h"
#include "bioArithUnaryExpression.h"
#include "bioBayesianResults.h"

class patNormalWichura ;
class patUniform ;

class bioArithBayes:  public bioExpression {
 public:
  bioArithBayes(bioExpressionRepository* rep,
		patULong par,
		vector<patULong> theBetas,
		patError*& err) ;
  
  virtual bioBayesianResults generateDraws(patError*& err)  ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual patBoolean isBayesian() const ;
  virtual void getNextDraw(patError*& err) = PURE_VIRTUAL ;
 protected:
  vector<patULong> betas ;
  vector<vector<patReal> > theDraws ;
  vector<patReal> betaValues ;
  vector<bioExpression*> betasExpr ;
  vector<patString> betaNames ;
  patNormalWichura* theNormal ;
  patUniform* theUniform ;

};

#endif
