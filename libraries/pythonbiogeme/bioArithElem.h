//-*-c++-*------------------------------------------------------------
//
// File name : bioArithElem.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Tue Jul 14 10:32:08  2009
//
//--------------------------------------------------------------------

#ifndef bioArithElem_h
#define bioArithElem_h


#include "bioExpression.h"

/*!
We have a dictionary of expressions, mapping values to expressions.
The value itself is first computed, and the associated expression is then evaluated.
*/
class bioArithElem : public bioExpression {

public:
  
  bioArithElem(bioExpressionRepository* rep,
	       patULong par,
	       patULong ind,
	       map<patULong,patULong> aDict,
	       patULong def,
	       patError*& err) ;
  ~bioArithElem() ;

public:
  virtual patBoolean isStructurallyZero() const ;
  virtual patString getOperatorName() const;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err)  ;
  virtual bioExpression* getDerivative(patULong aLiteralId, patError*& err) const ;
  virtual bioArithElem* getDeepCopy(bioExpressionRepository* rep,
				    patError*& err) const ;
  virtual bioArithElem* getShallowCopy(bioExpressionRepository* rep,
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
  patBoolean useDefault ;
  bioExpression* defaultExpression ;
  patULong defaultExpressionId ;
  bioExpression* indexCalculation ;
  patULong indexCalculationId ;
  // First argument: user ID
  // Second argument: pointer to the expression repository
  map<patULong,patULong> theDictionaryIds ;
  map<patULong,bioExpression*> theDictionary ;

  vector<bioExpression*> defaultDerivatives ;
  map<patULong,vector<bioExpression*> > theDerivDictionary ;

  //  vector<bioArithNamedExpression*> theNamedDefaultDerivative ;
  // map<patULong,vector<bioArithNamedExpression*> > theNamedDerivDictionary ;
  vector<patULong> theCompositeLiteralsIds ;

  patString variableWhereStored ;
  
};

#endif
