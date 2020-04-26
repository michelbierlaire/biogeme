//-*-c++-*------------------------------------------------------------
//
// File name : patArithEqual.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:59:31 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithEqual.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithEqual::patArithEqual(patArithNode* par,
			     patArithNode* left, 
			     patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithEqual::~patArithEqual() {
  
}

patArithNode::patOperatorType patArithEqual::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithEqual::getOperatorName() const {
  return("==") ;
}

patReal patArithEqual::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) == rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}



patReal patArithEqual::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithEqual::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithEqual* patArithEqual::getDeepCopy(patError*& err) {

  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (leftClone == NULL) {
      err = new patErrNullPointer("patArithNode") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL ;
    }
    if (rightClone == NULL) {
      err = new patErrNullPointer("patArithNode") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
  }

  patArithEqual* theNode = 
    new patArithEqual(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithEqual::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") == (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithEqual::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
