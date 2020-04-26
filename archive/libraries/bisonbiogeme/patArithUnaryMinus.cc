//-*-c++-*------------------------------------------------------------
//
// File name : patArithUnaryMinus.cc
// Author :    Michel Bierlaire
// Date :      Wed Nov 22 16:50:01 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithUnaryMinus.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithUnaryMinus::patArithUnaryMinus(patArithNode* par,
				       patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithUnaryMinus::~patArithUnaryMinus() {

}

patArithNode::patOperatorType patArithUnaryMinus::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithUnaryMinus::getOperatorName() const {
  return("-") ;
}

patReal patArithUnaryMinus::getValue(patError*& err) const {
  
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return -result ;
}

patReal patArithUnaryMinus::getDerivative(unsigned long index, 
					  patError*& err) const {
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  return -result ;
}


patString patArithUnaryMinus::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithUnaryMinus* patArithUnaryMinus::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithUnaryMinus* theNode = 
    new patArithUnaryMinus(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithUnaryMinus::getCppCode(patError*& err) {
  
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString vLeft = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  stringstream str ;
  str << "-(" << vLeft << ")" ;
  return patString(str.str()) ;
  
}

patString patArithUnaryMinus::getCppDerivativeCode(unsigned long index, patError*& err) {
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  if (leftChild->isDerivativeStructurallyZero(index, err)) {
    return patString("0.0") ;
  }

  stringstream str ;

  str << "- (" << leftChild->getCppDerivativeCode(index, err)  << " )" ;
  return patString(str.str()) ;
  
}
