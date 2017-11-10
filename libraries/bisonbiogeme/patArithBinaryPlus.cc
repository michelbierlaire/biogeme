//-*-c++-*------------------------------------------------------------
//
// File name : patArithBinaryPlus.cc
// Author :    Michel Bierlaire
// Date :      Wed Nov 22 17:30:32 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithBinaryPlus.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithBinaryPlus::patArithBinaryPlus(patArithNode* par,
				       patArithNode* left, 
				       patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithBinaryPlus::~patArithBinaryPlus() {

}

patArithNode::patOperatorType patArithBinaryPlus::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithBinaryPlus::getOperatorName() const {
  return("+") ;
}

patReal patArithBinaryPlus::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal left = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal right = rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal result = left + right ;

  return result ;
}

patReal patArithBinaryPlus::getDerivative(unsigned long index,
					  patError*& err) const {
  
//   DEBUG_MESSAGE("patArithBinaryPlus::getDerivative(" << index << ")") ;
//   DEBUG_MESSAGE(*this) ;
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal result = leftChild->getDerivative(index,err) +
    rightChild->getDerivative(index,err) ;
  //  DEBUG_MESSAGE(leftChild->getDerivative(index,err) << "+" << rightChild->getDerivative(index,err) << "=" << result) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patString patArithBinaryPlus::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithBinaryPlus* patArithBinaryPlus::getDeepCopy(patError*& err) {
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

  patArithBinaryPlus* theNode = 
    new patArithBinaryPlus(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithBinaryPlus::getCppCode(patError*& err) {
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
  str << "(" << vLeft << ") + (" << vRight << ")" ;
  return patString(str.str()) ;

}

patString patArithBinaryPlus::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString t1 = leftChild->getCppDerivativeCode(index, err) ;
  patString t2 = rightChild->getCppDerivativeCode(index, err) ;
  if (leftChild->isDerivativeStructurallyZero(index,err)) {
    if (rightChild->isDerivativeStructurallyZero(index,err)) {
      return patString("0.0") ;
    }
    return t2 ;
  }
  if (rightChild->isDerivativeStructurallyZero(index,err)) {
    return t1 ;
  }  

  stringstream str ;
  
  str << "( " << t1 << ") + ( " << t2 << ")" ;
  return patString(str.str()) ;

}
