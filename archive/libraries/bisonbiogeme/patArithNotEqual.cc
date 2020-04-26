//-*-c++-*------------------------------------------------------------
//
// File name : patArithNotEqual.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:09:51 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithNotEqual.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithNotEqual::patArithNotEqual(patArithNode* par,
				   patArithNode* left, 
				   patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithNotEqual::~patArithNotEqual() {
  
}

patArithNode::patOperatorType patArithNotEqual::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithNotEqual::getOperatorName() const {
  return("!=") ;
}

patReal patArithNotEqual::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) != rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithNotEqual::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithNotEqual::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithNotEqual* patArithNotEqual::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithNotEqual* theNode = 
    new patArithNotEqual(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithNotEqual::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") != (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithNotEqual::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
