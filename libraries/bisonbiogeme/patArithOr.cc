//-*-c++-*------------------------------------------------------------
//
// File name : patArithOr.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:12:19 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithOr.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithOr::patArithOr(patArithNode* par,
		       patArithNode* left, 
		       patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithOr::~patArithOr() {
  
}

patArithNode::patOperatorType patArithOr::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithOr::getOperatorName() const {
  return("||") ;
}

patReal patArithOr::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = ((leftChild->getValue(err)!=0.0) || 
		    (rightChild->getValue(err)!= 0.0)) ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithOr::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithOr::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithOr* patArithOr::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithOr* theNode = 
    new patArithOr(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}
patString patArithOr::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") + (" << vRight << "))" ;
  return patString(str.str()) ;

}


patString patArithOr::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
