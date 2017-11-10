//-*-c++-*------------------------------------------------------------
//
// File name : patArithGreaterEqual.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:25:45 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithGreaterEqual.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithGreaterEqual::patArithGreaterEqual(patArithNode* par,
					   patArithNode* left, 
					   patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithGreaterEqual::~patArithGreaterEqual() {
  
}

patArithNode::patOperatorType patArithGreaterEqual::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithGreaterEqual::getOperatorName() const {
  return(">=") ;
}

patReal patArithGreaterEqual::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) >= rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithGreaterEqual::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


patString patArithGreaterEqual::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithGreaterEqual* patArithGreaterEqual::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithGreaterEqual* theNode = 
    new patArithGreaterEqual(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithGreaterEqual::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") >= (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithGreaterEqual::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
