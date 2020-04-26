//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBinaryExpression.h
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 15:39:16 2011
//
//--------------------------------------------------------------------

#ifndef bioArithBinaryExpression_h
#define bioArithBinaryExpression_h


#include "bioExpression.h"

/*!
Abstract class for a binary expression
*/
class bioArithBinaryExpression: public bioExpression {

public:
  
  bioArithBinaryExpression(bioExpressionRepository* rep,
			   patULong par,
                           patULong lc,
			   patULong rc,
			   patError*& err) ;

  ~bioArithBinaryExpression() ;

public:
  virtual bioExpression* getLeftChild() const ;
  virtual bioExpression* getRightChild() const ;

  virtual patString getExpression(patError*& err) const ;
  virtual patULong getNumberOfOperations() const ;
  virtual patBoolean dependsOf(patULong aLiteralId) const ;
  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;
  virtual void simplifyZeros(patError*& err)  ;

  virtual void collectExpressionIds(set<patULong>* s) const ;
  virtual patString check(patError*& err) const  ;

protected:

  patULong leftChildId ;
  patULong rightChildId ;
  bioExpression* leftChild ;
  bioExpression* rightChild ;

};

#endif
