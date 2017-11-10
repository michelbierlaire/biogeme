//-*-c++-*------------------------------------------------------------
//
// File name : bioArithPrint.h
// Author :    Michel Bierlaire
// Date :      Tue May 11 07:08:49 2010
//
//--------------------------------------------------------------------

#ifndef bioArithPrint_h
#define bioArithPrint_h

#include "bioExpression.h"
#include "bioLiteral.h"
#include "bioIteratorInfo.h"
#include "bioSimulatedValues.h"

class patHybridMatrix ;
class patMultivariateNormal ;
/*!
Class implementing a node of the tree design to print the values of expressions
*/
class bioArithPrint : public bioExpression {

public:
  
  bioArithPrint(bioExpressionRepository* rep,
		patULong par,
		map<patString,patULong> theTerms,
		patString anIterator,
		patError*& err) ;

  ~bioArithPrint() ;

public:
  virtual bioExpression* getDerivative(patULong aLiteralId, 
				       patError*& err) const ;
  virtual patBoolean isSum() const  ;
  virtual patBoolean isProd() const  ;
  virtual patBoolean isSimulator() const ;
  // Simulate and perform sensitivity analysis
  void simulate(patHybridMatrix* varCovar, 
		vector<patString> betaNames, 
		bioExpression* weight,
		patError*& err) ;
  virtual patString getExpression(patError*& err) const ;
  virtual patString getOperatorName() const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
  virtual bioArithPrint* getDeepCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
  virtual bioArithPrint* getShallowCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
  virtual patString getExpressionString() const ;
  virtual patULong getNumberOfOperations() const ;
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;
  virtual patBoolean dependsOf(patULong aLiteralId) const  ;
  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;

  virtual patString getIncludeStatements(patString prefix) const ;
  virtual void simplifyZeros(patError*& err)  ;
  virtual void collectExpressionIds(set<patULong>* s) const  ;
  map<patString,bioSimulatedValues >* getSimulationResults()  ;
  virtual patString check(patError*& err) const  ;

private:
  map<patString,bioExpression*> theExpressions ;
  patString theIteratorName ;
  map<patString,patULong> theExpressionsId ;
  bioIteratorType theIteratorType ;

  map<patString,bioSimulatedValues > simulatedValues ;
  patMultivariateNormal* theRandomDraws ;
  
  vector<patBoolean> simulatedParam ;

};


#endif

