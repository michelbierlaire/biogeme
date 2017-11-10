//-*-c++-*------------------------------------------------------------
//
// File name : patNonLinearConstraint.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Mon Jan 20 09:19:47 2003
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patNonLinearConstraint.h"
#include "patValueVariables.h"
#include "patErrNullPointer.h"

patNonLinearConstraint::patNonLinearConstraint(patArithNode* node) :
  expression(node), dimension(0) {

}

patNonLinearConstraint::~patNonLinearConstraint() {

}

patReal patNonLinearConstraint::computeFunction(trVector* x,
						patBoolean* success,
						patError*& err) {
  if (x->size() != dimension) {
    *success = patFALSE ;
    stringstream str ;
    str << "Dimension should be " << dimension << " and not " << x->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patValueVariables::the()->setVariables(x) ;

  patReal result = expression->getValue(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;    
  }
  *success = patTRUE ;
  return result ;
}

patReal patNonLinearConstraint::computeFunctionAndDerivatives(trVector* x,
							      trVector* grad,
							      trHessian* hessian,
							      patBoolean* success,
							      patError*& err) {
  
  if (x->size() != dimension) {
    *success = patFALSE ;
    stringstream str ;
    str << "Dimension should be " << dimension << " and not " << x->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  if (grad->size() != dimension) {
    *success = patFALSE ;
    stringstream str ;
    str << "Dimension should be " << dimension << " and not " << grad->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patValueVariables::the()->setVariables(x) ;

  for (unsigned long i = 0 ; i < dimension ; ++i) {
    (*grad)[i] = expression->getDerivative(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal() ;    
    }
  }
  *success = patTRUE ;
  return expression->getValue(err) ;
  
}


trVector* patNonLinearConstraint::computeHessianTimesVector(trVector* x,
							    const trVector* v,
							    trVector* r,
							    patBoolean* success,
							    patError*& err)  {
  *success = patFALSE ;
  return r ;
}
  
trHessian* patNonLinearConstraint::getHessian() {
  return NULL ;
}

patBoolean patNonLinearConstraint::isGradientAvailable() const {
  return patTRUE ;
}

patBoolean patNonLinearConstraint::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

patBoolean patNonLinearConstraint::isHessianAvailable() const {
  return patFALSE ;
}

unsigned long patNonLinearConstraint::getDimension() const {
  return dimension ;
}

void patNonLinearConstraint::setDimension(unsigned long dim) {
  dimension = dim ;
}

void patNonLinearConstraint::setVariable(const patString& s, 
					 unsigned long i) {
  if (expression != NULL) {
    expression->setVariable(s,i) ;
  }
}

ostream& operator<<(ostream &str, const patNonLinearConstraint& x) {
  str << *x.expression ;
  return str ;
}

trHessian* patNonLinearConstraint::computeCheapHessian(trHessian* hessian,
						       patError*& err) {
  return NULL ;
}


patBoolean patNonLinearConstraint::isCheapHessianAvailable() {
  return patFALSE ;
}

void patNonLinearConstraint::generateCppCode(ostream& str, patError*& err) {
  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return ;
}

vector<patString>* patNonLinearConstraint::getLiterals(vector<patString>* listOfLiterals,
						       vector<patReal>* valuesOfLiterals,
						       patBoolean withRandom,
						       patError*& err) const {
  if (expression == NULL) {
    err = new patErrNullPointer("patArithNode") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  return expression->getLiterals(listOfLiterals,valuesOfLiterals,withRandom,err) ;
}
