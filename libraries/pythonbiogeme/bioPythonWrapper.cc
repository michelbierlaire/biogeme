//-*-c++-*------------------------------------------------------------
//
// File name : bioPythonWrapper.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 17 14:02:39 2009
//
//--------------------------------------------------------------------

#include <fstream>
#include "bioSample.h"
#include "patFileNames.h"
#include "bioPythonWrapper.h"
#include "patErrMiscError.h"
#include "bioParameters.h"
#include "bioIteratorInfoRepository.h"

bioPythonWrapper::bioPythonWrapper() {
}

bioPythonWrapper::~bioPythonWrapper() {

}


patReal bioPythonWrapper::computeFunction(trVector* x,
					  patBoolean* success,
					  patError*& err) {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

patReal bioPythonWrapper::computeFunctionAndDerivatives(trVector* x,
							trVector* grad,
							trHessian* hessian,
							patBoolean* success,
							patError*& err){
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return patReal() ;
}

trHessian* bioPythonWrapper::computeCheapHessian(trHessian* hessian,
						 patError*& err) {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return NULL ;
  
}

patBoolean bioPythonWrapper::isCheapHessianAvailable() {
  return patFALSE ;
}

trVector* bioPythonWrapper::computeHessianTimesVector(trVector* x,
						      const trVector* v,
						      trVector* r,
						      patBoolean* success,
						      patError*& err) {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return NULL ;
}

patBoolean bioPythonWrapper::isGradientAvailable() const {
  return patFALSE ;
}

patBoolean bioPythonWrapper::isHessianAvailable() const {
  return patFALSE ;
}

patBoolean bioPythonWrapper::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

unsigned long bioPythonWrapper::getDimension() const {
  return 0 ;
}


trVector bioPythonWrapper::getCurrentVariables(patError*& err) const {
  err = new patErrMiscError("This function should never be called. It must be overloaded") ;
  WARNING(err->describe()) ;
  return trVector() ;
}

patBoolean bioPythonWrapper::isUserBased() const {
  return patFALSE ;
}

void bioPythonWrapper::setSample(bioSample* s,patError*& err) {
  sample = s ;
  if (sample != NULL) {
    patULong nThreads = bioParameters::the()->getValueInt("numberOfThreads",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    patULong actualNumberOfThreads = bioIteratorInfoRepository::the()->generateSublistsForThreads(sample->size(),nThreads,err) ;
    if (actualNumberOfThreads < nThreads) {
      GENERAL_MESSAGE("Only " << actualNumberOfThreads << " threads are used due to the limited amount of data") ;
      bioParameters::the()->setValueInt("numberOfThreads",actualNumberOfThreads,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
}


void bioPythonWrapper::generateCppCode(ostream& str, patError*& err) {
  err = new patErrMiscError("Not implemented") ;
  WARNING(err->describe()) ;
  return ;
}
