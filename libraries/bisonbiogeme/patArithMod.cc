//-*-c++-*------------------------------------------------------------
//
// File name : patArithMod.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:36:49 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithMod.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithMod::patArithMod(patArithNode* par,
			 patArithNode* left, 
			 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithMod::~patArithMod() {
  
}

patArithNode::patOperatorType patArithMod::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithMod::getOperatorName() const {
  return("%") ;
}

patReal patArithMod::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  

  patReal result = int(leftChild->getValue(err)) % 
    int(rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  return result ;
}

patReal patArithMod::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithMod::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithMod* patArithMod::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithMod* theNode = 
    new patArithMod(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithMod::getCppCode(patError*& err) {
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
  str << "(" << vLeft << ") % (" << vRight << ")" ;
  return patString(str.str()) ;

}

patString patArithMod::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patString() ;

}
