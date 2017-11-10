//-*-c++-*------------------------------------------------------------
//
// File name : patArithNot.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:12:41 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithNot.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithNot::patArithNot(patArithNode* par,
			 patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithNot::~patArithNot() {

}

patArithNode::patOperatorType patArithNot::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithNot::getOperatorName() const {
  return("!") ;
}

patReal patArithNot::getValue(patError*& err) const {
  
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

  return ((result==0.0)? 1.0 : 0.0) ;
}

patReal patArithNot::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patString patArithNot::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}




patArithNot* patArithNot::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithNot* theNode = 
    new patArithNot(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithNot::getCppCode(patError*& err)  {
  
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
  str << "patBoolean(!(" << vLeft << "))" ;
  return patString(str.str()) ;
  
}

patString patArithNot::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
