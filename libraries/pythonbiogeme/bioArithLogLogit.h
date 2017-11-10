//-*-c++-*------------------------------------------------------------
//
// File name : bioArithLogLogit.h
// Author :    Michel Bierlaire
// Date :      Fri Jul 31 14:54:30 2015
//
//--------------------------------------------------------------------

#ifndef bioArithLogLogit_h
#define bioArithLogLogit_h


#include "bioExpression.h"

/*!
We have a dictionary of utilities, mapping values to utilities.
*/
class bioArithLogLogit : public bioExpression {

  friend class bioArithLogit ; 

public:
  
  bioArithLogLogit(bioExpressionRepository* rep,
		   patULong par,
		   patULong ind,
		   map<patULong,patULong> aDict,
		   map<patULong,patULong> avDict,
		   patBoolean ll,
		   patError*& err) ;
  ~bioArithLogLogit() ;

public:
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithLogLogit* getDeepCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual bioArithLogLogit* getShallowCopy(bioExpressionRepository* rep,
				     patError*& err) const ;
  virtual patString getExpression(patError*& err) const ;
  virtual patBoolean dependsOf(patULong aLiteralId) const ;
  virtual void simplifyZeros(patError*& err) ;
  virtual patULong getNumberOfOperations() const ;
  patULong lastIndex() const ;

  virtual patString getExpressionString() const ;
  patBoolean containsAnIntegral() const ;
  patBoolean containsAnIterator() const ;
  patBoolean containsAnIteratorOnRows() const ;
  patBoolean containsASequence() const ;
  virtual void collectExpressionIds(set<patULong>* s) const ;
  
  virtual bioFunctionAndDerivatives* getNumericalFunctionAndGradient(vector<patULong> literalIds, patBoolean computeHessian,				  patBoolean debugDerivatives,
patError*& err) ;

  virtual patString check(patError*& err) const  ;

private:
  patReal index ;
  bioExpression* indexCalculation ;
  patULong indexCalculationId ;
  // First argument: user ID
  // Second argument: pointer to the expression repository
  map<patULong,patULong> theDictionaryIds ;
  map<patULong,patULong> theAvailDictIds ;
  map<patULong,bioExpression*> theDictionary ;
  map<patULong,bioExpression*> theAvailDict ;
  vector<patULong> theCompositeLiteralsIds ;
  vector<bioFunctionAndDerivatives*> Vig ;
  vector<patReal> Vi ;
  patReal maxexp ;
  // If loglogit is patTRUE, the log of the probability is calculated
  patBoolean loglogit ;
  bioFunctionAndDerivatives logitresult ;

};

#endif
