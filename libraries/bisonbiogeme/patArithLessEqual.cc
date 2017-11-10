//-*-c++-*------------------------------------------------------------
//
// File name : patArithLessEqual.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:21:03 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithLessEqual.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithLessEqual::patArithLessEqual(patArithNode* par,
				     patArithNode* left, 
				     patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithLessEqual::~patArithLessEqual() {
  
}

patArithNode::patOperatorType patArithLessEqual::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithLessEqual::getOperatorName() const {
  return("<=") ;
}

patReal patArithLessEqual::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) <= rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithLessEqual::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithLessEqual::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithLessEqual* patArithLessEqual::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithLessEqual* theNode = 
    new patArithLessEqual(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithLessEqual::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") <= (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithLessEqual::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
