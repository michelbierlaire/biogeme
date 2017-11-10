//-*-c++-*------------------------------------------------------------
//
// File name : patArithLog.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:19:29 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include "patArithLog.h"
#include "patErrNullPointer.h"
#include "patDisplay.h"

patArithLog::patArithLog(patArithNode* par,
			 patArithNode* left) 
  : patArithNode(par,left,NULL) {

}

patArithLog::~patArithLog() {

}

patArithNode::patOperatorType patArithLog::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithLog::getOperatorName() const {
  return("log") ;
}

patReal patArithLog::getValue(patError*& err) const {
  
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

  return log(result) ;
}

patReal patArithLog::getDerivative(unsigned long index, 
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
  return deriv / value ;
}


patString patArithLog::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithLog* patArithLog::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithLog* theNode = 
    new patArithLog(NULL,leftClone) ;
  leftClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithLog::getCppCode(patError*& err) {
  
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
  str << "log(" << vLeft << ")" ;
  return patString(str.str()) ;
  
}

patString patArithLog::getCppDerivativeCode(unsigned long index, patError*& err) {

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
  str << "(" << deriv << ") / (" <<  value << ")" ;
  return patString(str.str()) ;

}
