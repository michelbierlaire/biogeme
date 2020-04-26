//-*-c++-*------------------------------------------------------------
//
// File name : patArithAttribute.cc
// Author :    Michel Bierlaire
// Date :      Wed Mar  5 13:30:23 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patArithAttribute.h"
#include "patValueVariables.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"

patArithAttribute::patArithAttribute(patArithNode* par)
  : patArithNode(par,NULL,NULL),index(patBadId ), calculatedExpression(NULL) 
{
  
}

patArithAttribute::~patArithAttribute() {

}

patArithNode::patOperatorType patArithAttribute::getOperatorType() const {
  return patArithNode::ATTRIBUTE_OP ;
}

patString patArithAttribute::getOperatorName() const {
  return(name) ;
}

patReal patArithAttribute::getValue(patError*& err) const {

  if (calculatedExpression != NULL) {
    patReal result = calculatedExpression->getValue(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return result ;
  }

  if ((index == patBadId) || !patValueVariables::the()->areAttributesAvailable()){
    patReal value =patValueVariables::the()->getValue(name,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return value ;
  }
  patReal value = patValueVariables::the()->getAttributeValue(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  return value ;

}

patReal patArithAttribute::getDerivative(unsigned long ind, 
					patError*& err) const {
  
  return 0.0 ;

}

patString patArithAttribute::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

void patArithAttribute::setName(const patString& n) {
  name = n ;
}

void patArithAttribute::setId(unsigned long id) {
  index = id ;
}

void patArithAttribute::replaceInLiterals(patString subChain, patString with) {
  replaceAll(&name,subChain,with) ;
}


patArithAttribute* patArithAttribute::getDeepCopy(patError*& err) {
  patArithAttribute* theNode = new patArithAttribute(NULL) ;
  theNode->leftChild = NULL ;
  theNode->rightChild = NULL ;
  theNode->name = name ;
  theNode->index = index ;
  return theNode ;
}

void patArithAttribute::setAttribute(const patString& s, unsigned long i) {
  if (s == name) {
    index = i ;
  }
}

void patArithAttribute::expand(patError*& err) {
  if (calculatedExpression != NULL) {
    return ;
  }

 patArithNode* ptrExpr = patModelSpec::the()->getVariableExpr(name,err);
 if (err != NULL) {
   WARNING(err->describe()); 
    return ;
 }
 if (ptrExpr != NULL) {
   if (name != ptrExpr->getOperatorName()) {
     calculatedExpression = ptrExpr ;
   }
 }
 

}


patString patArithAttribute::getCppCode(patError*& err) {
  if (calculatedExpression != NULL) {
    patString result = calculatedExpression->getCppCode(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString() ;
    }
    return result ;
  }
  
  if (index == patBadId) {
    stringstream str ;
    str << "Cannot generate code for attribute " << name ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return name ;
  }
  stringstream str ;
  str << "observation->attributes[" << index << "].value" ;
  return patString(str.str()) ;
}

patString patArithAttribute::getCppDerivativeCode(unsigned long index, patError*& err) {
  return patString("0.0") ;
}
