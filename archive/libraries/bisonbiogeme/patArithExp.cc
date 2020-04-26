//-*-c++-*------------------------------------------------------------
//
// File name : patArithExp.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:22:24 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patArithExp.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithExp::patArithExp(patArithNode* par,
			 patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithExp::~patArithExp() {

}

patArithNode::patOperatorType patArithExp::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithExp::getOperatorName() const {
  return("exp") ;
}

patReal patArithExp::getValue(patError*& err) const {
  
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
  
  if (!isfinite(result) || result >= patLogMaxReal::the()) {
    WARNING("Try to compute exp(" << result << ")") ;
    return patMaxReal ;
  }
  else {
    return exp(result) ;
  }
}

patReal patArithExp::getDerivative(unsigned long index,
				   patError*& err) const {
  
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patReal();
  }
  
  patReal value = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  patReal deriv = leftChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }

  if (!isfinite(value) || value >= patLogMaxReal::the()) {
    WARNING("Try to compute exp(" << value << ")") ;
    return patMaxReal ;
  } 
  else if (deriv * exp(value) >= patMaxReal) {
    WARNING("Try to compute exp(" << value << ")") ;
    return patMaxReal ;
  }
  else {
    return deriv * exp(value) ;
  }
}

patString patArithExp::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithExp* patArithExp::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithExp* theNode = 
    new patArithExp(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithExp::getCppCode(patError*& err) {
  
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
  str << "exp(" << vLeft << ")" ;
  return patString(str.str()) ;
  
}
patString patArithExp::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  patString value = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  patString deriv = leftChild->getCppDerivativeCode(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  if (deriv == "0.0" || leftChild->isDerivativeStructurallyZero(index,err)) {
    return patString("0.0") ;
  }
  stringstream str ;
  if (deriv == "1.0") {
    str << "exp(" << value << ")" ;
  }
  else {
    str << "( " << deriv << " ) * exp(" << value << ")" ;
  }
  return patString(str.str()) ; 


}
