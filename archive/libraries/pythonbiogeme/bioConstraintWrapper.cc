//-*-c++-*------------------------------------------------------------
//
// File name : bioConstraintWrapper.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Sun Oct 25 09:44:55 2009
//
//--------------------------------------------------------------------

#include "bioConstraintWrapper.h"


bioConstraintWrapper::bioConstraintWrapper() : trFunction() {

}

bioConstraintWrapper::~bioConstraintWrapper() {

}


patReal bioConstraintWrapper::computeFunction(trVector* x,
					      patBoolean* success,
					      patError*& err) {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patReal bioConstraintWrapper::computeFunctionAndDerivatives(trVector* x,
							    trVector* grad,
							    trHessian* hessian,
							    patBoolean* success,
							    patError*& err) {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return patReal() ;
}


trHessian* bioConstraintWrapper::computeCheapHessian(trHessian* hessian,
						     patError*& err) {
  return NULL ;
}

patBoolean bioConstraintWrapper::isCheapHessianAvailable() {
  return patFALSE ;
}

trVector* bioConstraintWrapper::computeHessianTimesVector(trVector* x,
							  const trVector* v,
							  trVector* r,
							  patBoolean* success,
							  patError*& err) {
  return NULL ;
}

patBoolean bioConstraintWrapper::isGradientAvailable()  const {
  return patTRUE ;
}

patBoolean bioConstraintWrapper::isHessianAvailable() const {
  return patTRUE ;
}

patBoolean bioConstraintWrapper::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

void bioConstraintWrapper::generateCppCode(ostream& str, patError*& err) {
  return ;
}

