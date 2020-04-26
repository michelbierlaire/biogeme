//-*-c++-*------------------------------------------------------------
//
// File name : patArithLess.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:18:02 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithLess.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithLess::patArithLess(patArithNode* par,
			   patArithNode* left, 
			   patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithLess::~patArithLess() {
  
}

patArithNode::patOperatorType patArithLess::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithLess::getOperatorName() const {
  return("<") ;
}

patReal patArithLess::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) < rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithLess::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithLess::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithLess* patArithLess::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithLess* theNode = 
    new patArithLess(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}
patString patArithLess::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") < (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithLess::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
