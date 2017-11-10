//-*-c++-*------------------------------------------------------------
//
// File name : patArithPower.cc
// Author :    Michel Bierlaire
// Date :      Thu Nov 23 15:55:39 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include "patArithPower.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patDisplay.h"

//
// Derivative formula
// f = u^v
// df / dx = u^v [ dv/dx ln u + (v/u) du/dx ] 
//


patArithPower::patArithPower(patArithNode* par,
			     patArithNode* left, 
			     patArithNode* right)
  : patArithNode(par,left,right) 
{
  
}

patArithPower::~patArithPower() {
  
}

patArithNode::patOperatorType patArithPower::getOperatorType() const {
  return patArithNode::BINARY_OP ;
}

patString patArithPower::getOperatorName() const {
  return("**") ;
}

patReal patArithPower::getValue(patError*& err) const {
  
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    WARNING(getInfo()) ;
    return patReal();
  }
  
  patReal l = leftChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  if (l == 0) {
    return 0 ;
  }
  patReal r = rightChild->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }


  patReal result = pow(l,r) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal();
  }
  if (isfinite(result) == 0) {
    stringstream str ;
    printLiterals(str,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    str << "Numerical problem in evaluating " << l << "**" << r << " within " 
	<< getInfo() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  return result ;
}

patReal patArithPower::getDerivative(unsigned long index, 
				     patError*& err) const {
  
  //  DEBUG_MESSAGE("patArithPower::getDerivative(" << index << ")") ;
  //  DEBUG_MESSAGE(*this) ;
  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    WARNING(getInfo()) ;
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




//   DEBUG_MESSAGE("leftValue=" << rightValue) ;
//   DEBUG_MESSAGE("rightValue=" << rightValue) ;
//   DEBUG_MESSAGE("leftDeriv=" << leftDeriv) ;
//   DEBUG_MESSAGE("rightDeriv=" << rightDeriv) ;

 
  patReal power =  pow(leftValue,rightValue) ;
  patReal result = 0 ;
  
  
  if (leftDeriv != 0.0) {
    if (rightValue == 1.0) {
      result += leftDeriv ;
    }
    else {
      result += power * leftDeriv * rightValue / leftValue ;
      if (isfinite(result) == 0) {
	stringstream str ;
	str << "Numerical problem in evaluating " << getInfo() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
    }
  }
  if (rightDeriv != 0.0) {
    if (leftValue != 0.0) {
      result += power * rightDeriv * log(leftValue) ;
      if (isfinite(result) == 0) {
	stringstream str ;
	str << "Numerical problem in evaluating " << getInfo() ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return patReal() ;
      }
    }
  }

  //  DEBUG_MESSAGE("result=" << result) ;
  return result ;
}

patString patArithPower::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

patArithPower* patArithPower::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithPower* theNode = 
    new patArithPower(NULL,leftClone,rightClone) ;
  leftClone->setParent(theNode) ;
  rightClone->setParent(theNode) ;
  return theNode ;
  
}

patString patArithPower::getGnuplot(patError*& err) const {

  patString result ;
  patString leftResult ;
  patString rightResult ;
      if (leftChild == NULL || rightChild == NULL) {
	err = new patErrNullPointer("patArithNode") ;
	WARNING(err->describe()) ;
	return patString();
      }

      leftResult = leftChild->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
      rightResult = rightChild->getGnuplot(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return patString();
      }
    
      result = "" ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += leftResult ;
      result += " " ;
      if (leftChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      result += " ** " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += "(" ;
      }
      result += " " ;
      result += rightResult ;
      result += " " ;
      if (rightChild->getOperatorType() == BINARY_OP) {
	result += ")" ;
      }
      return result ;

}


patString patArithPower::getCppCode(patError*& err) {
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
  if (vLeft == "0.0") {
    return patString("0.0") ;
  }
  stringstream str ;
  if (vRight == "1.0") {
    return vLeft ;
  }
  str << "pow(" << vLeft << "," << vRight << ")" ;
  return patString(str.str()) ;
  
}

patString patArithPower::getCppDerivativeCode(unsigned long index, patError*& err) {

  if (leftChild == NULL || rightChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    WARNING(getInfo()) ;
    return patString();
  }
  
  patString leftValue = leftChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  patString rightValue = rightChild->getCppCode(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  patString leftDeriv = leftChild->getCppDerivativeCode(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  patString rightDeriv = rightChild->getCppDerivativeCode(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }

  if (leftValue == "0.0") {
    return patString("0.0") ;
  }


  stringstream p ;
  p << "pow(" << leftValue << "," << rightValue << ")" ;
 
  patBoolean firstTerm(!leftChild->isDerivativeStructurallyZero(index,err)) ;
  patBoolean secondTerm(!rightChild->isDerivativeStructurallyZero(index,err)) ;

  
  patBoolean something(patFALSE) ;
  stringstream str ;
  if (firstTerm) {
    if (leftDeriv != "0.0" && rightValue != "0.0") {
      str <<  p.str() << " * (" << leftDeriv << ") * (" << rightValue << ") / (" << leftValue << ")" ;
      something = patTRUE ;
      if (secondTerm) {
	str << " + " ;
      }
    }
  }
  if (secondTerm) {
    if (rightDeriv != "0.0") {
      str <<  p.str() << " * (" << rightDeriv << ") * log(" << leftValue << ")" ;
      something = patTRUE ;
    }
  }
  
  if (!something) {
    return patString("0.0") ;
  }
  //  DEBUG_MESSAGE("result=" << result) ;
  return patString(str.str()) ;


}
