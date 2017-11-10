//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnaryExpression.h
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 15:19:16 2011
//
//--------------------------------------------------------------------

#ifndef bioArithUnaryExpression_h
#define bioArithUnaryExpression_h


#include "bioExpression.h"

/*!
Class implementing a node of the tree representing a sum expression
*/
class bioArithUnaryExpression : public bioExpression {

public:
  
  bioArithUnaryExpression(bioExpressionRepository* rep, 
			  patULong par,
			  patULong c,
			  patError*& err) ;

  ~bioArithUnaryExpression() ;

public:
  virtual bioExpression* getChild() const ;

  virtual patString getExpression(patError*& err) const ;
  virtual patULong getNumberOfOperations() const ;
  virtual patBoolean dependsOf(patULong aLiteralId) const ;

  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;

  virtual void simplifyZeros(patError*& err)  ;

  virtual void collectExpressionIds(set<patULong>* s) const  ;
  virtual patString check(patError*& err) const  ;

protected:

  patULong childId ;
  bioExpression* child ;

};

#endif

