//-*-c++-*------------------------------------------------------------
//
// File name : bioArithUnaryExpression.cc
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 15:28:44 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioArithUnaryExpression.h"
#include "patErrNullPointer.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"
#include "patErrMiscError.h"

bioArithUnaryExpression::bioArithUnaryExpression(bioExpressionRepository* rep,
						 patULong par,
						 patULong c,
						 patError*& err) :
  bioExpression(rep,par), childId(c) {

  child = theRepository->getExpression(childId) ;
  if (child == NULL) {
    DEBUG_MESSAGE("Expression: " << getExpression(err)) ;
    stringstream str ;
    str << "No expression with id " << childId ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  //child->setParent(getId(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  relatedExpressions.push_back(child) ;
}

bioArithUnaryExpression::~bioArithUnaryExpression() {

}

bioExpression* bioArithUnaryExpression::getChild() const {
  return child ;
}

patString bioArithUnaryExpression::getExpression(patError*& err) const {
  if (child == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString childResult = child->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString result =  getOperatorName() ;
  result += '(' ;
  result += childResult ;
  result += ')' ;
  return result ;
}

patULong bioArithUnaryExpression::getNumberOfOperations() const {
  if (child == NULL) {
    return 0 ;
  }
  patULong result = child->getNumberOfOperations() ;
  return (1 + result) ;
}

patBoolean bioArithUnaryExpression::dependsOf(patULong aLiteralId) const {
  if (child == NULL) {
    return patFALSE ;
  }  
  return (child->dependsOf(aLiteralId)) ;
}


patBoolean bioArithUnaryExpression::containsAnIterator() const {
  return child->containsAnIterator() ;
}

patBoolean bioArithUnaryExpression::containsAnIteratorOnRows() const {
  return child->containsAnIteratorOnRows() ;
}

patBoolean bioArithUnaryExpression::containsAnIntegral() const {
  return child->containsAnIntegral() ;
}

patBoolean bioArithUnaryExpression::containsASequence() const {
  return child->containsASequence() ;
}

void bioArithUnaryExpression::simplifyZeros(patError*& err) {
  bioExpression* newChild(NULL) ;
  if (child != NULL) {
    if (child->isStructurallyZero()) {
      newChild = theRepository->getZero() ;
    }
    else if (child->isConstant()) {
      patReal value = child->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
      newChild = new bioArithConstant(theRepository,patBadId,value) ;
    }
    else {
      child->simplifyZeros(err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
    }
  }
  if (newChild != NULL) {
    child = newChild ;
    childId = child->getId() ;
  }
}



void bioArithUnaryExpression::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  child->collectExpressionIds(s) ;
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->collectExpressionIds(s) ;
  }
}



patString bioArithUnaryExpression::check(patError*& err) const {
  stringstream str ;
  if (child->getId() != childId) {
    str << "Incompatible IDS for children: " << child->getId() << " and " << childId ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
  return patString() ;
}

