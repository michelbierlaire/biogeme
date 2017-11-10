//-*-c++-*------------------------------------------------------------
//
// File name : patArithDeriv.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Mon Feb 20 15:02:02 2006
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithDeriv.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"

patArithDeriv::patArithDeriv(patArithNode* par,
			     patArithNode* mainExpression,
			     patString param) : 
  patArithNode(par,mainExpression,NULL),
  parameter(param), first(patTRUE), index(patBadId)
{
}

patArithDeriv::~patArithDeriv() {

}

patArithNode::patOperatorType patArithDeriv::getOperatorType() const {
  return patArithNode::UNARY_OP ;
}

patString patArithDeriv::getOperatorName() const {
  return("$DERIV") ;
}

patReal patArithDeriv::getValue(patError*& err) const {
  
  //    DEBUG_MESSAGE("Compute deriv. of " <<leftChild->getExpression(err) << " wrt " << parameter) ;
  patReal result = leftChild->getDerivative(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return result ;
  //   DEBUG_MESSAGE("result = " << result) ;
}
  
patReal patArithDeriv::getDerivative(unsigned long index, patError*& err) const {
  stringstream str ;
  str << "Operator " << getOperatorName() 
      << " can be used only in the specification of derivatives" ;
  err = new patErrMiscError(str.str()) ;
  WARNING(err->describe()) ;
  return patReal() ;
}
  
patString patArithDeriv::getExpression(patError*& err) const {


  patString result ;
  patString leftResult ;
  if (leftChild == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return patString();
  }
  
  leftResult = leftChild->getExpression(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patString();
  }
  
  result = "$DERIV" ;
  result += "(" ;
  result += leftResult ;
  result += " , " ;
  result += parameter ;
  result += " )" ;
  return result ;
}
  
patArithDeriv* patArithDeriv::getDeepCopy(patError*& err) {
  patArithNode* leftClone(NULL) ;
  patArithNode* rightClone(NULL) ;
  if (leftChild != NULL) {
    leftClone = leftChild->getDeepCopy(err) ;
  }
  if (rightChild != NULL) {
    rightClone = rightChild->getDeepCopy(err) ;
  }

  patArithDeriv* theNode = new patArithDeriv(NULL,leftClone,parameter) ;
  leftClone->setParent(theNode) ;
  return theNode ;
}

void patArithDeriv::computeParamId(patError*& err) {
  // *************************
  // WARNING
  // It seems that there is a confusion wrt index vs id
  // Computing the derivative requires the id of the beta and not the index
  // *************************
  if (first) {
    index =  patModelSpec::the()->getBetaId(parameter,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    first = patFALSE ;
  }

}

patString patArithDeriv::getCppCode(patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return patString() ;

}
patString patArithDeriv::getCppDerivativeCode(unsigned long index, patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return patString() ;

}
