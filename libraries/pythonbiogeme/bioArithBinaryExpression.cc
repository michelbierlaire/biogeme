//-*-c++-*------------------------------------------------------------
//
// File name : bioArithBinaryExpression.cc
// Author :    Michel Bierlaire
// Date :      Wed Apr 13 15:41:50 2011
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "patErrNullPointer.h"
#include "bioArithBinaryExpression.h"
#include "bioArithConstant.h"
#include "bioExpressionRepository.h"
#include "patErrMiscError.h"

bioArithBinaryExpression::bioArithBinaryExpression(bioExpressionRepository* rep,
						   patULong par,
						   patULong lc,
						   patULong rc,
						   patError*& err) 
  : bioExpression(rep,par), leftChildId(lc), rightChildId(rc) {

  if (leftChildId != patBadId) {
    leftChild = theRepository->getExpression(lc) ;
    if (leftChild != NULL) {
      //leftChild->setParent(getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else {
      stringstream str ;
      str << "No expression with ID " << lc ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;

      return ;
    }
  }
  if (rightChildId != patBadId) {
    rightChild = theRepository->getExpression(rc) ;
    if (rightChild != NULL) {
      //rightChild->setParent(getId(),err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else {
      stringstream str ;
      str << "No expression with ID " << lc ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
    return ;
    }
  }
  relatedExpressions.push_back(leftChild) ;
  relatedExpressions.push_back(rightChild) ;
}

bioArithBinaryExpression::~bioArithBinaryExpression() {

}

bioExpression* bioArithBinaryExpression::getLeftChild() const {
  return leftChild ;
}
bioExpression* bioArithBinaryExpression::getRightChild() const {
  return rightChild ;
}

patString bioArithBinaryExpression::getExpression(patError*& err) const {
    if (leftChild == NULL || rightChild == NULL) {
      err = new patErrNullPointer("bioExpression") ;
      WARNING(err->describe()) ;
      return patString();
    }
    
    patString leftResult = leftChild->getExpression(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
    patString rightResult = rightChild->getExpression(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString();
    }
    
    patString result = "" ;
    if (!leftChild->isLiteral()) {
      result += "( " ;
    }
    result += leftResult ;
    if (!leftChild->isLiteral()) {
      result += " )" ;
    }
    result += " " ;
    result += getOperatorName() ;
    result += " " ;
    if (!rightChild->isLiteral()) {
      result += "( " ;
    }
    result += rightResult ;
    if (!rightChild->isLiteral()) {
      result += " )" ;
    }
    return result ;
}


patULong bioArithBinaryExpression::getNumberOfOperations() const {
  if (leftChild == NULL || rightChild == NULL) { 
    return 0 ;
  }
  patULong l = leftChild->getNumberOfOperations() ;
  patULong r = rightChild->getNumberOfOperations() ;
  return (1+l+r) ;
}

 patBoolean bioArithBinaryExpression::dependsOf(patULong aLiteralId) const {
  if (leftChild->dependsOf(aLiteralId)) {
    return patTRUE ;
  }
  return rightChild->dependsOf(aLiteralId) ;
}



patBoolean bioArithBinaryExpression::containsAnIterator() const {
  if (leftChild != NULL) {
    if (leftChild->containsAnIterator()) {
      return patTRUE ;
    }
  }
  if (rightChild != NULL) {
    return rightChild->containsAnIterator() ;
  }
  else {
    return patFALSE ;
  }
}

patBoolean bioArithBinaryExpression::containsAnIteratorOnRows() const {
  if (leftChild != NULL) {
    if (leftChild->containsAnIteratorOnRows()) {
      return patTRUE ;
    }
  }
  if (rightChild != NULL) {
    return rightChild->containsAnIteratorOnRows() ;
  }
  else {
    return patFALSE ;
  }
}

patBoolean bioArithBinaryExpression::containsAnIntegral() const {
  if (leftChild != NULL) {
    if (leftChild->containsAnIntegral()) {
      return patTRUE ;
    }
  }
  if (rightChild != NULL) {
    return rightChild->containsAnIntegral() ;
  }
  else {
    return patFALSE ;
  }

}

patBoolean bioArithBinaryExpression::containsASequence() const {
  if (leftChild != NULL) {
    if (leftChild->containsASequence()) {
      return patTRUE ;
    }
  }
  if (rightChild != NULL) {
    return rightChild->containsASequence() ;
  }
  else {
    return patFALSE ;
  }
}

void bioArithBinaryExpression::simplifyZeros(patError*& err) {
  bioExpression* newLeftChild(NULL) ;
  bioExpression* newRightChild(NULL) ;
  if (leftChild != NULL) {
    if (leftChild->isStructurallyZero()) {
      newLeftChild = theRepository->getZero() ;
    }
    else if (leftChild->isConstant()) {
      patReal value = leftChild->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
      newLeftChild = new bioArithConstant(theRepository,getId(),value) ;
    }
    else {
      leftChild->simplifyZeros(err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
    }
  }
  if (rightChild != NULL) {
    if (rightChild->isStructurallyZero()) {
      newRightChild = theRepository->getZero() ;
    }
    else if (rightChild->isConstant()) {
      patReal value = rightChild->getValue(patFALSE, patLapForceCompute, err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
      newRightChild = new bioArithConstant(theRepository,getId(),value) ;
    }
    else {
      rightChild->simplifyZeros(err) ;
      if (err != NULL) {
        WARNING(err->describe()) ;
        return ;
      }
    }
  }
  if (newLeftChild != NULL) {
    leftChildId = newLeftChild->getId() ;
    leftChild = newLeftChild ;
  }
  if (newRightChild != NULL) {
    rightChildId = newRightChild->getId() ;
    rightChild = newRightChild ;
  }

}





void bioArithBinaryExpression::collectExpressionIds(set<patULong>* s) const {
  s->insert(getId()) ;
  leftChild->collectExpressionIds(s) ;
  rightChild->collectExpressionIds(s) ;
  for (vector<bioExpression*>::const_iterator i = relatedExpressions.begin() ;
       i != relatedExpressions.end() ;
       ++i) {
    (*i)->collectExpressionIds(s) ;
  }
}



patString bioArithBinaryExpression::check(patError*& err) const {
  stringstream str ;
  if (leftChild->getId() != leftChildId) {
    str << "Incompatible IDS for children: " << leftChild->getId() << " and " << leftChildId ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
  if (rightChild->getId() != rightChildId) {
    str << "Incompatible IDS for children: " << rightChild->getId() << " and " << rightChildId ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
  return patString() ;
}

