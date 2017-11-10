//-*-c++-*------------------------------------------------------------
//
// File name : patArithAbs.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:19:29 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patArithAbs.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithAbs::patArithAbs(patArithNode* par,
			 patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithAbs::~patArithAbs() {

}

patArithNode::patOperatorType patArithAbs::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithAbs::getOperatorName() const {
  return("abs") ;
}

patReal patArithAbs::getValue(patError*& err) const {
  
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

  return ((result>=0)?result:-result) ;
}

patReal patArithAbs::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithAbs::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithAbs* patArithAbs::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithAbs* theNode = 
    new patArithAbs(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithAbs::getCppCode(patError*& err) {

  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString result = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  stringstream str ;
  str << "patAbs(" << result << ")" ;
  return patString(str.str()) ;

}

patString patArithAbs::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative available") ;
  WARNING(err->describe()) ;
  return patString() ;

}
