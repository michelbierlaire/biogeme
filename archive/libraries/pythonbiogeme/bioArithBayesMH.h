//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBayesMH.h
// Author :    Michel Bierlaire
// Date :      Tue Jul 31 16:08:15 2012
//
//--------------------------------------------------------------------

#ifndef bioArithBayesMH_h
#define bioArithBayesMH_h

#include "bioLiteral.h"
#include "bioIteratorInfo.h"
#include "bioArithBayes.h"

/*!
Class implementing a node of the tree designed for the Metropolis Hastings algorithm, to draw from a complex density
*/
class bioArithBayesMH : public bioArithBayes {

public:
  
  bioArithBayesMH(bioExpressionRepository* rep,
		  patULong par,
		  vector<patULong> theBetas,
		  patULong theDensity,
		  patULong theWarmup,
		  patULong theSteps,
		  patError*& err) ;

  ~bioArithBayesMH() ;

public:
  // Implementation of pure virtual function from bioExpression
  virtual patString getOperatorName() const ;
  virtual patString getExpression(patError*& err) const ;
  virtual bioArithBayesMH* getDeepCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
  virtual bioArithBayesMH* getShallowCopy(bioExpressionRepository* rep, 
				     patError*& err) const ;
  virtual patBoolean dependsOf(patULong aLiteralId) const  ;
  virtual patString getExpressionString() const ;


  virtual patULong getNumberOfOperations() const ;

  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;

  virtual void simplifyZeros(patError*& err)  ;

  virtual void collectExpressionIds(set<patULong>* s) const  ;
  virtual patString check(patError*& err) const  ;


  void initializeMarkovProcess(patError*& err) ;

  patBoolean updateBetas(patError*& err) ;
  virtual void getNextDraw(patError*& err)  ;
private:
  patULong warmup ;
  patULong steps ;
  patReal rho ;
  patULong total ;
  patULong accept ;
  patReal logCurrentValue ;
  bioExpression* densityExpression ;
};

#endif

