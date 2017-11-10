//-*-c++-*------------------------------------------------------------
//
// File name : patArithMult.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:48:30 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithMult.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithMult::patArithMult(patArithNode* par,
			   patArithNode* left, 
			   patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithMult::~patArithMult() {
  
}

patArithNode::patOperatorType patArithMult::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithMult::getOperatorName() const {
  return("*") ;
}

patReal patArithMult::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal left = leftChild->getValue(err) ;
  if (err != NULL) {
    DEBUG_MESSAGE("Current expression: " << *this) ;
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal right = rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal result = left * right ;

  return result ;
}

patReal patArithMult::getDerivative(unsigned long index,
				    patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal leftValue = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal rightValue = rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal leftDeriv = leftChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal rightDeriv = rightChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal result = leftDeriv * rightValue + leftValue * rightDeriv ;
  return result ;
}

patString patArithMult::getExpression(patError*&
 err) const {
  return patArithNode::getExpression(err) ;
}

patArithMult* patArithMult::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithMult* theNode = 
    new patArithMult(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithMult::getCppCode(patError*& err) {
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
  if ((vLeft == "0.0") || vRight == "0.0") {
    return patString("0.0") ;
  }
  else {
    stringstream str ;
    str << "(" << vLeft << ") * (" << vRight << ")" ;
    return patString(str.str()) ;
  }

}


patString patArithMult::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }

  patString v1 = leftChild->getCppCode(err) ;
  patString v2 = rightChild->getCppCode(err) ;
  patString d1 = leftChild->getCppDerivativeCode(index, err) ;
  patString d2 = rightChild->getCppDerivativeCode(index, err) ;

  stringstream str ;
  
  if (leftChild->isDerivativeStructurallyZero(index,err)) {
    if (rightChild->isDerivativeStructurallyZero(index,err)) {
      return patString("0.0") ;
    }
    str << "(" << v1 << ") * (" << d2 << ")" ;
    return patString(str.str()) ;
  }
  if (rightChild->isDerivativeStructurallyZero(index,err)) {
    if (d1 == "0.0" || v2 == "0.0") {
      return patString("0.0") ;
    }
    else {
      str << "(" << d1 << ") * (" << v2 << ")" ;
      return patString(str.str()) ;
    }
  } 
  patBoolean nullTerm1((v1 == "0.0") || (d2 == "0.0")) ;
  patBoolean nullTerm2((d1 == "0.0") || (v2 == "0.0")) ;

  if (nullTerm1) {
    if (nullTerm2) {
      return patString("0.0") ;
    }
    else {
      str << "(" << d1 << ") * (" << v2 << ")" ;
      return patString(str.str()) ;
    }
  }
  else {
    if (nullTerm2) {
      str << "(" << v1 << ") * (" << d2 << ")"  ;
      return patString(str.str()) ;
    }
    else{
      str << "(" << v1 << ") * (" << d2 << ") + (" << d1 << ") * (" << v2 << ")" ;
      return patString(str.str()) ;
    }
  }
}

