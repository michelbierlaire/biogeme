//-*-c++-*------------------------------------------------------------
//
// File name : patArithConstant.cc
// Author :    Michel Bierlaire
// Date :      Wed Nov 22 22:18:43 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patArithConstant.h"
#include "patDisplay.h"

patArithConstant::patArithConstant(patArithNode* par)
  : patArithNode(par,NULL,NULL)
{
  
}

patArithConstant::~patArithConstant() {

}

patArithNode::patOperatorType patArithConstant::getOperatorType() const {
  return patArithNode::CONSTANT_OP ;
}

patString patArithConstant::getOperatorName() const {
  patError* err = NULL;
  return(getExpression(err)) ;
}

patReal patArithConstant::getValue(patError*& err) const {
  
  return value ;
}

patReal patArithConstant::getDerivative(unsigned long index, patError*& err) const {
  return 0.0 ;
}


patString patArithConstant::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

void patArithConstant::setValue(patReal v) {
  value = v ; 
}

patArithConstant* patArithConstant::getDeepCopy(patError*& err) {
  patArithConstant* theNode = new patArithConstant(NULL) ;
  theNode->leftChild = NULL ;
  theNode->rightChild = NULL ;
  theNode->value = value ;
  return theNode ;
}

patString patArithConstant::getCppCode(patError*& err) {

  stringstream str ;
  str << value ;
  return patString(str.str()) ;

}

patString patArithConstant::getCppDerivativeCode(unsigned long index, patError*& err) {
  return patString("0.0") ;
}
