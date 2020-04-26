//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayesMean.h
// Author :    Michel Bierlaire
// Date :      Thu Oct 18 16:21:02 2012
//
//--------------------------------------------------------------------

#ifndef bioArithBayesMean_h
#define bioArithBayesMean_h

#include "bioArithBayes.h"

class patMultivariateNormal ;
/*
Class implementing a node of the tree designed for drawing from the
posterior of the mean of a normal variable knowing realizations and
variance. See Train (2003)  p. 304 step 1. 
*/

class bioArithBayesMean : public bioArithBayes {
 public:
  bioArithBayesMean(bioExpressionRepository* rep,
		    patULong par,
		    vector<patULong> means,
		    vector<patULong> realizations,
		    vector<vector<patULong> > varcovar,
		    patError*& err) ;
  
  ~bioArithBayesMean() ;
  
 public:
  // Implementation of pure virtual function from bioExpression
   patString getOperatorName() const ;
   patString getExpression(patError*& err) const ;
   bioArithBayesMean* getDeepCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
   bioArithBayesMean* getShallowCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
   patBoolean dependsOf(patULong aLiteralId) const  ;
   patString getExpressionString() const ;

  virtual patULong getNumberOfOperations() const ;

  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;

  virtual void simplifyZeros(patError*& err)  ;

  virtual void collectExpressionIds(set<patULong>* s) const  ;
  virtual patString check(patError*& err) const  ;


  // Implementation of pure virtual function from bioArithBayes

   void getNextDraw(patError*& err) ;
  
private:
  void prepareTheDraws(patError*& err) ;
private:
  vector<patULong> theRealizationsExpression ;
  vector<vector<patULong> > theVarcovarExpression ;
  patString individualId ;
  patMultivariateNormal* theRandomDraws ;

  patVariables* mu ;
  patHybridMatrix* sigma ;
};

#endif
