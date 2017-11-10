//-*-c++-*------------------------------------------------------------
//
// File name : patArithAnd.cc
// Author :    Michel Bierlaire
// Date :      Fri Nov 24 09:15:39 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithAnd.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithAnd::patArithAnd(patArithNode* par,
			 patArithNode* left, 
			 patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithAnd::~patArithAnd() {
  
}

patArithNode::patOperatorType patArithAnd::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithAnd::getOperatorName() const {
  return(" * ") ;
}

patReal patArithAnd::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  

  patReal result = ((leftChild->getValue(err)!=0.0) &&
		    (rightChild->getValue(err)!= 0.0)) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithAnd::getDerivative(unsigned long index, patError*& err) const {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


patString patArithAnd::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithAnd* patArithAnd::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithAnd* theNode = 
    new patArithAnd(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}


patString patArithAnd::getCppCode(patError*& err) {
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
  str << "patBoolean((" << vLeft << ") * (" << vRight << "))" ;
  return patString(str.str()) ;

}

patString patArithAnd::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("No derivative for boolean functions") ;
  WARNING(err->describe()) ;
  return patString() ;

}
