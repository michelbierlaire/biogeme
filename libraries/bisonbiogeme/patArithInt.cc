//-*-c++-*------------------------------------------------------------
//
// File name : patArithInt.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:41:32 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patArithInt.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithInt::patArithInt(patArithNode* par,
			 patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithInt::~patArithInt() {

}

patArithNode::patOperatorType patArithInt::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithInt::getOperatorName() const {
  return("int") ;
}

patReal patArithInt::getValue(patError*& err) const {
  
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return (rint(result)) ;
}

patReal patArithInt::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithInt::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithInt* patArithInt::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithInt* theNode = 
    new patArithInt(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithInt::getCppCode(patError*& err) {
  
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString vLeft = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  stringstream str ;
  str << "rint(" << vLeft << ")" ;
  return patString(str.str()) ;
  
}

patString patArithInt::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patString() ;

}
