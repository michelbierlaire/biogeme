//-*-c++-*------------------------------------------------------------
//
// File name : trFunction.cc
// Author :    Michel Bierlaire
// Date :      Mon Apr  9 08:58:13 2001
//
//--------------------------------------------------------------------

#include "patMath.h"
#include "patErrNullPointer.h"
#include "trFunction.h"
#include "patFileExists.h"
#include "patLoopTime.h"

trFunction::~trFunction() {

}
trVector* trFunction::computeFinDiffGradient(trVector* x,
					     trVector* g,
					     patBoolean* success,
					     patError*& err) {
  

  if (err != NULL) {
    WARNING(err->describe());
    return NULL;
  }

  if (x == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (g == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  if (x->size() != g->size()) {
    stringstream str ;
    str << "Incompatible sizes: " << x->size() << " and " << g->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
  }
  // gradient by finite differences


  patReal fc = computeFunction(x,success,err) ;

  if (err != NULL) {
    WARNING(err->describe());
    return NULL;
  }
  if (!(*success)) {
    return NULL ;
  }


  trVector xplus = *x ;
  patReal sqrteta = pow(patEPSILON, 0.5);

  unsigned long dim = getDimension() ;

  if (x->size() != dim) {
    stringstream str ;
    str << "Problem dimension is " << dim << ", not " << x->size() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }


  for (patULong j = 0; j < dim ; ++j) {

    patReal stepsizej = sqrteta*
      patMax(patAbs(xplus[j]), patReal(1.0))*patSgn(xplus[j]);

    patReal tempj = xplus[j];
    
    xplus[j] += stepsizej;
    stepsizej = xplus[j] - tempj;
    patReal fj = computeFunction(&xplus, success,err);
    if (err != NULL) {
      WARNING(err->describe());
      return NULL;
    }
    if (!(*success)) {
      return NULL ;
    }
    
    (*g)[j] = ((fj - fc)/stepsizej); 
    xplus[j] = tempj;
    
    
  }

  return g ;

  
}

trHessian* trFunction::computeFinDiffHessian(trVector* x,
					     trHessian* theHessian,
					     patBoolean* success,
					     patError*& err) {
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return NULL ;
  }
  
  unsigned long dim = getDimension() ;
  if (dim != x->size()) {
    stringstream str ;
    str << "Incompatible size between function (" << getDimension() 
	<< ") and variables (" << x->size() << ")"  ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }


  trVector gradc(dim) ;
  computeFunctionAndDerivatives(x,&gradc,NULL,success,err) ;
//   DEBUG_MESSAGE("x=" << *x) ;
//   DEBUG_MESSAGE("g=" << gradc) ;
  if (err != NULL) {
    WARNING(err->describe());
    return NULL;
  }
  if (!(*success)) {
    DEBUG_MESSAGE("PROBLEM IN COMPUTING GRADIENT") ;
    return NULL ;
  }


  vector<trVector> h(dim,trVector(dim)) ;

  trVector xplus(*x) ;
  patReal sqrteta = pow(patEPSILON, 0.5);

  GENERAL_MESSAGE("Compute hessian by finite difference") ;
  GENERAL_MESSAGE("You can interrupt it by creating a file named " << stopFileName) ;
  patReal currentStepProgress =  0.0 ;

  patDisplay::the().initProgressReport("Compute hessian by finite difference",
					dim) ;

  trVector gradj(dim) ;

  patLoopTime loopTime(dim) ;

  for (patULong j = 0; j < dim ; ++j) {
    if (patFileExists()(stopFileName)) {
      WARNING("Computation interrupted by the user with the file " 
	      << stopFileName) ;
      *success =patFALSE ;
      return NULL ;
    }
      
    patReal currentProgress = 100.0 * j / dim ;
    if (j != 0 && currentProgress >= currentStepProgress) {
      loopTime.setIteration(j) ;
      GENERAL_MESSAGE(currentStepProgress << "%\t" << loopTime) ;
      currentStepProgress += 10.0 ;
    }
    if (!patDisplay::the().updateProgressReport(j)) {
      patDisplay::the().terminateProgressReport() ;
      err = new patErrMiscError("Interrupted by the user") ;
      WARNING(err->describe()) ;
      return NULL ;
    }

    patReal stepsizej = sqrteta*
      patMax(patAbs(xplus[j]), patReal(1.0))*patSgn(xplus[j]);

    patReal tempj = xplus[j];
    
    xplus[j] += stepsizej;
    stepsizej = xplus[j] - tempj;
    computeFunctionAndDerivatives(&xplus,&gradj,NULL, success,err);
    if (err != NULL) {
      WARNING(err->describe());
      return NULL;
    }
    if (!(*success)) {
      DEBUG_MESSAGE("PROBLEM IN COMPUTING GRADIENT (2)") ;
      return NULL ;
    }

    h[j] = (1.0/stepsizej) * (gradj - gradc) ;

//     DEBUG_MESSAGE("h[" << j << "]=" << h[j]) ;
    xplus[j] = tempj;
  }

  for (unsigned long i = 0 ; i < dim ; ++i) {
    for (unsigned long j = i ; j < dim ; ++j) {
      theHessian->setElement(i,j,(h[i][j]+h[j][i])/2.0,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return NULL;
      }
    }
  }
  patDisplay::the().terminateProgressReport() ;

  return theHessian ;
}



void trFunction::setStopFileName(patString f) {
  stopFileName = f ;
}


