//-*-c++-*------------------------------------------------------------
//
// File name : patArithMax.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:28:44 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithMax.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patMath.h"
#include "patDisplay.h"

patArithMax::patArithMax(patArithNode* par,
			 patArithNode* left, 
			 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithMax::~patArithMax() {
  
}

patArithNode::patOperatorType patArithMax::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithMax::getOperatorName() const {
  return("max") ;
}

patReal patArithMax::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  

  patReal result = patMax(leftChild->getValue(err),
			  rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  return result ;
}

patReal patArithMax::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


patString patArithMax::getExpression(patError*& err) const {


  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString() ;
  }

  patString leftResult = leftChild->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
  patString rightResult = rightChild->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString() ;
  }
    
  patString result = "max(" ;
  result += leftResult ;
  result += "," ; 
  result += rightResult ;
  result += ")" ;
  return result ;
}

patArithMax* patArithMax::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithMax* theNode = 
    new patArithMax(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithMax::getCppCode(patError*& err) {
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
  str << "patMax(" << vLeft << "," << vRight << ")" ;
  return patString(str.str()) ;
  
}

patString patArithMax::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patString() ;

}
