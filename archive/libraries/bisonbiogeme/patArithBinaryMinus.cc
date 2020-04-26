//-*-c++-*------------------------------------------------------------
//
// File name : patArithBinaryMinus.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:45:30 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithBinaryMinus.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithBinaryMinus::patArithBinaryMinus(patArithNode* par,
					 patArithNode* left, 
					 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithBinaryMinus::~patArithBinaryMinus() {
  
}

patArithNode::patOperatorType patArithBinaryMinus::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithBinaryMinus::getOperatorName() const {
  return("-") ;
}

patReal patArithBinaryMinus::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getValue(err) - rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}


patReal patArithBinaryMinus::getDerivative(unsigned long index,
					   patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getDerivative(index,err) - 
    rightChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patString patArithBinaryMinus::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithBinaryMinus* patArithBinaryMinus::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithBinaryMinus* theNode = 
    new patArithBinaryMinus(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithBinaryMinus::getCppCode(patError*& err) {
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString vLeft = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  patString vRight = rightChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  stringstream str ;
  str << "(" << vLeft << ") - (" << vRight << ")" ;
  return patString(str.str()) ;

}

patString patArithBinaryMinus::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  stringstream str ;
  
  patString t1 = leftChild->getCppDerivativeCode(index, err) ;
  patString t2 = rightChild->getCppDerivativeCode(index, err) ;
  if (leftChild->isDerivativeStructurallyZero(index,err)) {
    if (rightChild->isDerivativeStructurallyZero(index,err)) {
      return patString("0.0") ;
    }
    str << "- ( " << t2 << ")" ;
    return patString(str.str()) ;
  }
  if (rightChild->isDerivativeStructurallyZero(index,err)) {
    return t1 ;
  }  
  str << "( " << t1 << ") - ( " << t2 << ")" ;
  return patString(str.str()) ;

}

