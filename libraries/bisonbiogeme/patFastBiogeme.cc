//-*-c++-*------------------------------------------------------------
//
// File name : patFastBiogeme.cc
// Author :    Michel Bierlaire
// Date :      Tue Aug 14 10:43:54 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <fstream>
#include "patVersion.h"
#include "patFileNames.h"
#include "patFastBiogeme.h"
#include "patErrMiscError.h"
patFastBiogeme::patFastBiogeme() {

}
patFastBiogeme::~patFastBiogeme() {

}

patReal patFastBiogeme::getFunction(trVector* x,
				    patBoolean* success,
				    patError*& err) {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

trVector* patFastBiogeme::getGradient(trVector* x,
				      trVector* grad,
				      patBoolean* success,
				      patError*& err) {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return NULL ;
}

patReal patFastBiogeme::getFunctionAndGradient(trVector* x,
					       trVector* grad,
					       patBoolean* success,
					       patError*& err){
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

trHessian* patFastBiogeme::patFastBiogeme::computeHessian(patVariables* x,
							  trHessian& hessian,
							  patBoolean* success,
							  patError*& err) {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return NULL ;

}
 
trHessian* patFastBiogeme::getCheapHessian(trVector* x,
					   trHessian& hessian,
					   patBoolean* success,
					   patError*& err) {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return NULL ;
  
}

patBoolean patFastBiogeme::isCheapHessianAvailable() {
  return patFALSE ;
}

trVector patFastBiogeme::getHessianTimesVector(trVector* x,
					       const trVector* v,
					       patBoolean* success,
					       patError*& err) {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return trVector() ;
}

patBoolean patFastBiogeme::isGradientAvailable() const {
  return patFALSE ;
}

patBoolean patFastBiogeme::isHessianAvailable() const {
  return patFALSE ;
}

patBoolean patFastBiogeme::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

unsigned long patFastBiogeme::getDimension() const {
  WARNING("This function should never be called") ;
  return 0 ;
}


trVector patFastBiogeme::getCurrentVariables(patError*& err) const {
  err = new patErrMiscError("This function should never be called") ;
  WARNING(err->describe()) ;
  return trVector() ;
}

patBoolean patFastBiogeme::isUserBased() const {
  return patFALSE ;
}

void patFastBiogeme::setSample(patSample* s) {
  sample = s ;
}


void patFastBiogeme::generateCppCode(ostream& str, patError*& err) {
  err = new patErrMiscError("Not implemented") ;
  WARNING(err->describe()) ;
  return ;
}
