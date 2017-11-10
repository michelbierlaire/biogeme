//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMultinaryExpression.h
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 19:50:05 2011
//
//--------------------------------------------------------------------

#ifndef bioArithMultinaryExpression_h
#define bioArithMultinaryExpression_h


#include "bioExpression.h"

/*!
Abstract class for a multinary expressions
*/

class bioArithMultinaryExpression: public bioExpression {

public:
  
  bioArithMultinaryExpression(bioExpressionRepository* rep,
			      patULong par,
			      vector<patULong> l, 
			      patError*& err) ;
  ~bioArithMultinaryExpression() ;

public:
  virtual vector<patULong> getChildrenIds() const ;
  virtual bioExpression* getChild(patULong index, patError*& err) const ;

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

  vector<bioExpression*> listOfChildren ;
  vector<patULong> listOfChildrenIds ;

};

#endif
