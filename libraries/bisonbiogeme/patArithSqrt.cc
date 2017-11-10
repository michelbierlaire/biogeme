//-*-c++-*------------------------------------------------------------
//
// File name : patArithSqrt.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:16:37 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patArithSqrt.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithSqrt::patArithSqrt(patArithNode* par,
			   patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithSqrt::~patArithSqrt() {

}

patArithNode::patOperatorType patArithSqrt::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithSqrt::getOperatorName() const {
  return("sqrt") ;
}

patReal patArithSqrt::getValue(patError*& err) const {
  
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
  return pow(result,0.5) ;
}

patReal patArithSqrt::getDerivative(unsigned long index, 
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

  return (0.5 * pow(value,-0.5) * deriv) ;

  
}


patString patArithSqrt::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithSqrt* patArithSqrt::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithSqrt* theNode = 
    new patArithSqrt(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithSqrt::getCppCode(patError*& err) {
  
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
  str << "pow(" << vLeft << ",0.5)" ;
  return patString(str.str()) ;
  
}

patString patArithSqrt::getCppDerivativeCode(unsigned long index, patError*& err) {
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
  
  stringstream str ;
  str << "(0.5 * pow(" << value << ",-0.5) * " << deriv << ")" ;
  return patString(str.str()) ;

}
