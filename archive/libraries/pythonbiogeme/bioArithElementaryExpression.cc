//-*-c++-*------------------------------------------------------------
//
// File name : bioArithElementaryExpression.cc
// Author :    Michel Bierlaire
// Date :      Sat Apr 16 14:28:39 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioArithElementaryExpression.h"
#include "patErrNullPointer.h"

bioArithElementaryExpression::bioArithElementaryExpression(bioExpressionRepository* rep,
							   patULong par) :
  bioExpression(rep,par) {

}

bioArithElementaryExpression::~bioArithElementaryExpression() {

}

patString bioArithElementaryExpression::getExpression(patError*& err) const {
  return  getOperatorName() ;
}

patULong bioArithElementaryExpression::getNumberOfOperations() const {
  return 0 ;
}



patBoolean bioArithElementaryExpression::containsAnIterator() const {
  return patFALSE ;
}

patBoolean bioArithElementaryExpression::containsAnIteratorOnRows() const {
  return patFALSE ;
}

patBoolean bioArithElementaryExpression::containsAnIntegral() const {
  return patFALSE ;
}

patBoolean bioArithElementaryExpression::containsASequence() const {
  return patFALSE ;
}

bioExpression* bioArithElementaryExpression::getExpressionFromId(patULong theId) {
  if (theId == getId()) {
    return this ;
  }
  return NULL ;
}

void bioArithElementaryExpression::simplifyZeros(patError*& err) {
  return ;
}

void bioArithElementaryExpression::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
}

patString bioArithElementaryExpression::check(patError*& err) const {
  return patString() ;
}
