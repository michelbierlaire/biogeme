//-*-c++-*------------------------------------------------------------
//
// File name : patArithVariable.cc
// Author :    Michel Bierlaire
// Date :      Wed Nov 22 22:30:03 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patDisplay.h"
#include "patArithVariable.h"
#include "patValueVariables.h"
#include "patErrNullPointer.h"
#include "patModelSpec.h"

patArithVariable::patArithVariable(patArithNode* par)
  : patArithNode(par,NULL,NULL),index(patBadId ), xIndex(patBadId), calculatedExpression(NULL) 
{
  
}

patArithVariable::~patArithVariable() {

}

patArithNode::patOperatorType patArithVariable::getOperatorType() const {
  return patArithNode::VARIABLE_OP ;
}

patString patArithVariable::getOperatorName() const {
  return(name) ;
}

patReal patArithVariable::getValue(patError*& err) const {
  
  if (calculatedExpression != NULL) {
    patReal result = calculatedExpression->getValue(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;
    }
    return result ;
  }

  if ((index == patBadId) || !patValueVariables::the()->areVariablesAvailable()) {
    //    DEBUG_MESSAGE("index=" << index) ;
    patReal result = patValueVariables::the()->getValue(name,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patReal() ;
    }
    return result ;
  }

  patReal value =  patValueVariables::the()->getVariable(index,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return value ;

}

patReal patArithVariable::getDerivative(unsigned long ind, 
					patError*& err) const {
  
//   DEBUG_MESSAGE("patArithVariable::getDerivative(" << ind << ")") ;
//   DEBUG_MESSAGE(*this) ;
//   DEBUG_MESSAGE("index=" << index) ;

  if (index == patBadId) {
    return 0 ;
  }
  return (ind == index)?1.0:0.0 ;

}

patString patArithVariable::getExpression(patError*& err) const {
  return patArithNode::getExpression(err) ;
}

void patArithVariable::setName(patString n) {
  name = n ;
}

void patArithVariable::setVariable(const patString& s, unsigned long i) {
  if (name == s) {
    index = i ;
  }
}

void patArithVariable::replaceInLiterals(patString subChain, patString with) {
  replaceAll(&name,subChain,with) ;
}

patArithVariable* patArithVariable::getDeepCopy(patError*& err) {
  patArithVariable* theVar = new patArithVariable(NULL) ;
  theVar->leftChild = NULL ;
  theVar->rightChild = NULL ;
  theVar->name = name ;
  theVar->index = index ;
  return theVar ;
}

void patArithVariable::expand(patError*& err) {
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

patString patArithVariable::getCppCode(patError*& err) {

  if (xIndex == patBadId) {
    patBoolean found ;
    patBetaLikeParameter theParam = patModelSpec::the()->getParameterFromId(index,
									    &found);
    if (!found) {
      stringstream str ;
      str << "Parameter ID " << index << " not found" ;
      err = new patErrMiscError(str.str()); 
      return patString() ;
    }
    xIndex = theParam.index ;
    value  = theParam.defaultValue ; 
  }

  if (calculatedExpression != NULL) {
    patString result = calculatedExpression->getCppCode(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patString() ;
    }
    return result ;
  }

  stringstream str ;
  if (xIndex == patBadId) {
    str << value ;
  }
  else {
    str << "(*x)[" << xIndex << "]" ;
  }
  return patString(str.str()) ;


}

patBoolean patArithVariable::isDerivativeStructurallyZero(unsigned long theIndex, patError*& err) {

  if (xIndex == patBadId) {
    patBoolean found ;
    patBetaLikeParameter theParam = patModelSpec::the()->getParameterFromId(index,
									    &found);
    if (!found) {
      stringstream str ;
      str << "Parameter ID " << index << " not found" ;
      err = new patErrMiscError(str.str()); 
      return patFALSE ;
    }
    xIndex = theParam.index ;
    
  }
  map<unsigned long,patBoolean>::iterator found = derivStructurallyZero.find(xIndex) ;
  if (found != derivStructurallyZero.end()) {
    return found->second ;
  }

  derivStructurallyZero[theIndex] = (theIndex != xIndex) ;
  return (theIndex != xIndex) ;
}

patString patArithVariable::getCppDerivativeCode(unsigned long ind, patError*& err) {
  if (xIndex == patBadId) {
    patBoolean found ;
    patBetaLikeParameter theParam = patModelSpec::the()->getParameterFromId(index,
									    &found);
    if (!found) {
      stringstream str ;
      str << "Parameter ID " << index << " not found" ;
      err = new patErrMiscError(str.str()); 
      return patString() ;
    }
    xIndex = theParam.index ;
    value = theParam.defaultValue ;
    
  }
  if (xIndex == ind) {
    return "1.0" ;
  }
  return "0.0" ;

}
