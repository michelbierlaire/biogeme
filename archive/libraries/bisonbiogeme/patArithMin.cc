//-*-c++-*------------------------------------------------------------
//
// File name : patArithMin.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:33:10 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithMin.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patMath.h"
#include "patDisplay.h"

patArithMin::patArithMin(patArithNode* par,
			 patArithNode* left, 
			 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithMin::~patArithMin() {
  
}

patArithNode::patOperatorType patArithMin::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithMin::getOperatorName() const {
  return("min") ;
}

patReal patArithMin::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = patMin(leftChild->getValue(err),
			  rightChild->getValue(err)) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithMin::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithMin::getExpression(patError*& err) const {


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
    
  patString result = "min(" ;
  result += leftResult ;
  result += "," ; 
  result += rightResult ;
  result += ")" ;
  return result ;
}

patArithMin* patArithMin::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithMin* theNode = 
    new patArithMin(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithMin::getCppCode(patError*& err) {
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
  str << "patMin(" << vLeft << "," << vRight << ")" ;
  return patString(str.str()) ;
  
}


patString patArithMin::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patString() ;

}
