//-*-c++-*------------------------------------------------------------
//
// File name : patArithGreater.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:23:20 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithGreater.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithGreater::patArithGreater(patArithNode* par,
				 patArithNode* left, 
				 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithGreater::~patArithGreater() {
  
}

patArithNode::patOperatorType patArithGreater::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithGreater::getOperatorName() const {
  return(">") ;
}

patReal patArithGreater::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = (leftChild->getValue(err) > rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithGreater::getDerivative(unsigned long index, 
				       patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithGreater::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithGreater* patArithGreater::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithGreater* theNode = 
    new patArithGreater(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithGreater::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") > (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithGreater::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
