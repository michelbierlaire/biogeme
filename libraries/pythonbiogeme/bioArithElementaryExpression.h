//-*-c++-*------------------------------------------------------------
//
// File name : bioArithElementaryExpression.h
// Author :    Michel Bierlaire
// Date :      Sat Apr 16 14:27:34 2011
//
//--------------------------------------------------------------------

#ifndef bioArithElementaryExpression_h
#define bioArithElementaryExpression_h


#include "bioExpression.h"

/*!
Class implementing a node of the tree representing an elementary expression, without any chil
*/
class bioArithElementaryExpression : public bioExpression {

public:
  
  bioArithElementaryExpression(bioExpressionRepository* rep, patULong par) ;

  ~bioArithElementaryExpression() ;

public:

  virtual patULong getNumberOfOperations() const ;

  virtual patBoolean containsAnIterator() const ;
  virtual patBoolean containsAnIteratorOnRows() const ;
  virtual patBoolean containsAnIntegral() const ;
  virtual patBoolean containsASequence() const ;

  virtual patString getExpression(patError*& err) const ;
  virtual bioExpression* getExpressionFromId(patULong theId) ;
  virtual void simplifyZeros(patError*& err)  ;

  virtual void collectExpressionIds(set<patULong>* s) const ;
  virtual patString check(patError*& err) const  ;

};

#endif
