//-*-c++-*------------------------------------------------------------
//
// File name : patArithDivide.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:50:50 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithDivide.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

patArithDivide::patArithDivide(patArithNode* par,
			       patArithNode* left, 
			       patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithDivide::~patArithDivide() {
  
}

patArithNode::patOperatorType patArithDivide::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithDivide::getOperatorName() const {
  return("/") ;
}

patReal patArithDivide::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  if (leftChild->getValue(err) == 0.0) {
    return 0.0 ;
  }  
  // Require better numerical check. To be done later on...
  if (rightChild->getValue(err) == 0.0) {
    err = new patErrMiscError("Divide by zero error") ;
    WARNING(err->describe()) ;
    return patReal();
  }

  patReal result = leftChild->getValue(err) / rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  return result ;
}

patReal patArithDivide::getDerivative(unsigned long index, 
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

  
  // Require better numerical check. To be done later on...
  if (rightValue == 0.0) {
    err = new patErrMiscError("Divide by zero error") ;
    WARNING(err->describe()) ;
    return patReal();
  }

  patReal result = 
    (rightValue * leftDeriv - leftValue * rightDeriv) / 
    (rightValue * rightValue) ;

  return result ;
}

patString patArithDivide::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithDivide* patArithDivide::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithDivide* theNode = 
    new patArithDivide(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}


patString patArithDivide::getCppCode(patError*& err) {
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
  str << "(" << vLeft << ") / (" << vRight << ")" ;
  return patString(str.str()) ;

}

patString patArithDivide::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }

  patString vLeft = leftChild->getCppCode(err) ;
  patString vRight = rightChild->getCppCode(err) ;
  patString dLeft = leftChild->getCppDerivativeCode(index, err) ;
  patString dRight = rightChild->getCppDerivativeCode(index, err) ;

  stringstream str ;

  if (leftChild->isDerivativeStructurallyZero(index,err)) {
    if (rightChild->isDerivativeStructurallyZero(index,err)) {
      str << "0.0" ;
    }
    else {
      if (vLeft == "0.0" || dRight == "0.0") {
	str << "0.0" ;
      }
      else
	str << "( - " << vLeft << " * " << dRight << ") / (" << vRight << " * " << vRight <<")" ;
    }
  }
  else {
    if (rightChild->isDerivativeStructurallyZero(index,err)) {
      if (vRight == "0.0" || dLeft == "0.0") {
	str << "0.0" ;
      }
      else {
	str << "(" << vRight << " * " << dLeft << ") / (" << vRight << " * " << vRight <<")" ;
      }
      
    }
    else {
      str << "(" << vRight << " * " << dLeft << " - " << vLeft << " * " << dRight << ") / (" << vRight << " * " << vRight <<")" ;
      
    }

  }
  return patString(str.str()) ;

}
