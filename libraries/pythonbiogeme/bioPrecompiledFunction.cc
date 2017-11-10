//-*-c++-*------------------------------------------------------------
//
// File name : bioPrecompiledFunction.cc
// Author :    Michel Bierlaire
// Date :      Mon Apr 18 18:09:01 2011
//
//--------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iterator>
#include "patMath.h"
#include "trParameters.h"
#include "bioPrecompiledFunction.h"
#include "trParameters.h"
#include "bioLiteralRepository.h"
#include "bioParameters.h"
#include "bioIteratorInfoRepository.h"
#include "bioExpression.h"
#include "patErrNullPointer.h"
#include "bioExpressionRepository.h"
#include "patLargeNumbers.h"
#include "patDisplay.h"

#include "patLap.h"

#include <pthread.h> 
void *computeFunctionForThread( void *ptr );
void *computeFunctionAndGradientForThread( void *ptr );



bioPrecompiledFunction::bioPrecompiledFunction(bioExpressionRepository* rep, 
					       patULong expId,
					       patError*& err) 
: theExpressionRepository(bioParameters::the()->getValueInt("numberOfThreads")), 
  theExpressionId(expId),
  bhhh(bioParameters::the()->getTrParameters(err),
       bioLiteralRepository::the()->getNumberOfEstimatedParameters()), 
  threadGrad(bioParameters::the()->getValueInt("numberOfThreads"),
	     trVector(bioLiteralRepository::the()->getNumberOfEstimatedParameters())), 
  threadBhhh(bioParameters::the()->getValueInt("numberOfThreads"),
	     trHessian(bioParameters::the()->getTrParameters(err),
		       bioLiteralRepository::the()->getNumberOfEstimatedParameters())), 
  threadSuccess(bioParameters::the()->getValueInt("numberOfThreads"),patFALSE),
  threadHessian(bioParameters::the()->getValueInt("numberOfThreads"),
		trHessian(bioParameters::the()->getTrParameters(err),
			  bioLiteralRepository::the()->getNumberOfEstimatedParameters() )), 
  threadChildren(bioParameters::the()->getValueInt("numberOfThreads")),
  threadSpans(bioParameters::the()->getValueInt("numberOfThreads")) 
{

  DEBUG_MESSAGE("Repository of size: " << rep->getNbrOfExpressions()) ;
  bioExpression* theExpression = rep->getExpression(theExpressionId) ;
  theExpression->checkMonteCarlo(patFALSE,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
  }

  patULong nThreads = bioParameters::the()->getValueInt("numberOfThreads") ;
  if (theExpression->containsAnIterator() && nThreads > 1 && !theExpression->isSumIterator()) {
    err = new patErrMiscError("Multithreading can be used only if the expression is a sum over the sample. If you want to work with this expression, reduce the number of threads to 1: BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = \"1\"") ;
    WARNING(err->describe()) ;
    return  ;
  }

  bhhh.setToZero() ;
  betaIds = bioLiteralRepository::the()->getBetaIds(patFALSE, err)  ;

  
  for (patULong i = 0 ; i < theExpressionRepository.size() ; ++i) {
    if (rep != NULL) {
      bioExpressionRepository* theCopy = rep->getCopyForThread(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
      }
      //theCopy->check(err) ;
      if (err != NULL) {
	//DEBUG_MESSAGE("Original: " << rep->printList()) ;
	//DEBUG_MESSAGE("Copy: " << theCopy->printList()) ;
	WARNING(err->describe()) ;
      }
      theCopy->getExpression(theExpressionId) ;
      // theExpr->prepareDerivatives(betaIds,err) ;
      // if (err != NULL) {
      // 	WARNING(err->describe()) ;
      // }

      theExpressionRepository[i] = theCopy ;
    }
    else {
      theExpressionRepository[i] = NULL ;
    }
  }

  threads = new pthread_t[nThreads];
  input = new bioThreadArg[nThreads];

}

bioPrecompiledFunction::~bioPrecompiledFunction() {

}

patReal bioPrecompiledFunction::computeFunction(trVector* x,
						patBoolean* success,
						patError*& err) {


  short nThreads = bioParameters::the()->getValueInt("numberOfThreads") ;

  patLap::next();
  //DEBUG_MESSAGE("start computeFunction lap:" << l);
  bioLiteralRepository::the()->setBetaValues(*x,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal();
  }
  trVector theBetas = bioLiteralRepository::the()->getBetaValues() ; 
  patReal logLike = 0.0 ;
  
  if (theExpressionRepository[0] == NULL) {
    err = new patErrNullPointer("bioExpressionRepository") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal();
  }

  bioExpression* theExpression = theExpressionRepository[0]->getExpression(theExpressionId) ;
  if (theExpression == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal();
  }
  
  if (theExpression->isSumIterator()) {
  
    for(int i = 0; i < nThreads; ++i) {
      input[i].threadId = i ;
      input[i].grad = NULL ;
      input[i].hessian = NULL ;
      input[i].bhhh = NULL ;
      input[i].success = &(threadSuccess[i]) ;
      input[i].sample = sample ;
      input[i].result = 0.0 ;
      threadSpans[i] = bioIteratorInfoRepository::the()->getSublist(i,err) ; 
      if (err != NULL) {
	WARNING(err->describe()) ;
	*success = patFALSE ;
	return patReal();
      }
      input[i].subsample = &(threadSpans[i]) ;
      input[i].theExpressionRepository = theExpressionRepository[i] ;
      input[i].theExpressionId = theExpressionId ;
      
      if(pthread_create( &threads[i], NULL, computeFunctionForThread, (void*) &input[i])) {
	err = new patErrMiscError("Error creating Threads! Exit!") ;
	WARNING(err->describe()) ;
	*success = patFALSE ;
	return 0.0;
      }

    }

    *success = patTRUE ;
    logLike = 0.0 ;
    patBoolean maxreal(patFALSE) ;
    for(int i=0; i<nThreads; ++i) {

      pthread_join( threads[i], NULL);
      if (input[i].err != NULL) {
	WARNING(input[i].err->describe()) ;
	err = input[i].err ;
	*success = patFALSE ;
  	return patReal() ;
      }
      if (!input[i].success) {
	*success = patFALSE ;
      }
      if (!maxreal) {
      	if (input[i].result == -patMaxReal) {
      	  logLike = patMaxReal ;
      	  maxreal = patTRUE ;
      	}
      	else {
      	  logLike -= input[i].result ;
      	}
      }
    }
  } else {
    // The expression is not an iterator. Threads should not be used.
    patReal result = theExpression->getValue(patFALSE, patLap::get(), err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal();
    }
    *success = patTRUE ;
    logLike = -result ;
    
  }
    
  //DEBUG_MESSAGE("computeFunction done")

  if (!patFinite(logLike)) {
    return patMaxReal ;
  }

  return logLike ;
}
  


patReal bioPrecompiledFunction::computeFunctionAndDerivatives(trVector* x,
							      trVector* grad,
							      trHessian* h,
							      patBoolean* success,
							      patError*& err) {
  patBoolean computeHessian = (h != NULL) ;
  
  if (grad == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;
  }

  short nThreads = bioParameters::the()->getValueInt("numberOfThreads") ;
  *success = patTRUE ;
  bhhh.setToZero() ;
  if (computeHessian) {
    h->setToZero() ;
  }
  bioLiteralRepository::the()->setBetaValues(*x,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal();
  }
  trVector theBetas = bioLiteralRepository::the()->getBetaValues() ; 
  patReal logLike = 0.0 ;
  
  fill(grad->begin(),grad->end(),0.0) ; 

  bioExpression* theExpression = theExpressionRepository[0]->getExpression(theExpressionId) ;
  if (theExpression == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal();
  }
 
  if (theExpression->isSumIterator()) {
    for(int i=0; i<nThreads; i++) {
      input[i].threadId = i ;
      fill(threadGrad[i].begin(),threadGrad[i].end(),0.0) ;
      input[i].grad = &(threadGrad[i]) ;
      if (computeHessian) {
	threadHessian[i].setToZero() ;    
	input[i].hessian = &(threadHessian[i]) ;
      }
      else {
	input[i].hessian = NULL ;
	
      }
      threadBhhh[i].setToZero() ;    
      input[i].bhhh = &(threadBhhh[i]) ;
      input[i].success = &(threadSuccess[i]) ;
      input[i].sample = sample ;
      input[i].result = 0.0 ;
      input[i].literalIds = betaIds ;
      threadSpans[i] = bioIteratorInfoRepository::the()->getSublist(i,err) ;  
      if (err != NULL) {
	WARNING(err->describe()) ;
	*success = patFALSE ;
	return patReal();
      }
      input[i].subsample = &(threadSpans[i]) ;
      input[i].theExpressionRepository = theExpressionRepository[i] ;
      input[i].theExpressionId = theExpressionId ;
      
      if(pthread_create( &threads[i], NULL, computeFunctionAndGradientForThread, (void*) &input[i])) {
	err = new patErrMiscError("Error creating Threads! Exit!") ;
	WARNING(err->describe()) ;
	*success = patFALSE ;
	return 0.0;
      }
    }
    
    void *tmp;
    
    patBoolean maxreal(patFALSE) ;
    logLike = 0.0 ; 
    for(int i=0; i<nThreads; ++i) {
      pthread_join( threads[i], &tmp);
      if (input[i].err != NULL) {
	WARNING(input[i].err->describe()) ;
	err = input[i].err ;
	*success = patFALSE ;
	return patReal() ;
      }
      if (!input[i].success) {
	*success = patFALSE ;
      }
      if (!maxreal) {
	if (input[i].result == -patMaxReal) {
	  logLike = patMaxReal ;
	  maxreal = patTRUE ;
	}
	else {
	  
	  logLike -= input[i].result ;
	  (*grad) -= threadGrad[i] ;
	  bhhh.add(1.0,threadBhhh[i],err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	*success = patFALSE ;
	return 0.0;
	  }
	  if (computeHessian) {
	    h->add(-1.0,threadHessian[i],err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      *success = patFALSE ;
	      return 0.0;
	    }
	  }
	}
      }
    }
  }
  else {
    // The expression is not an iterator. Threads should not be used.
    patBoolean debugDeriv(patFALSE) ;
#ifdef DEBUG
    debugDeriv = (bioParameters::the()->getValueInt("debugDerivatives",err) != 0) ;
#endif
    if (err != NULL) {
      WARNING(err->describe()) ;
      *success = patFALSE ;
      return patReal() ;
    }
    bioFunctionAndDerivatives* result = theExpression->getNumericalFunctionAndGradient(betaIds,computeHessian,debugDeriv,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patReal();
    }
    logLike = -result->theFunction;
    *grad = -result->theGradient ;
    if (computeHessian && result->theHessian != NULL) {
      result->theHessian->changeSign() ;
      *h = *(result->theHessian) ;
    }
    
  }

  if (!patFinite(logLike)) {
    return patMaxReal ;
  }
  
  return logLike ;

}



trHessian* bioPrecompiledFunction::computeCheapHessian(trHessian* hessian,
						       patError*& err) {

  if (hessian == NULL) {
    err = new patErrNullPointer("trHessian") ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  *hessian = bhhh ;
  return hessian ;
}

patBoolean bioPrecompiledFunction::isCheapHessianAvailable() {
  return patTRUE ;
}

trVector* bioPrecompiledFunction::computeHessianTimesVector(trVector* x,
							    const trVector* v,
							    trVector* r,
							    patBoolean* success,
							    patError*& err) {
  err = new patErrMiscError("Not implemented") ;
  WARNING(err->describe()) ;
  return NULL ;
}

patBoolean bioPrecompiledFunction::isGradientAvailable() const {
  return patTRUE ;
}

patBoolean bioPrecompiledFunction::isHessianAvailable()  const {
  patError* err(NULL) ;
  int h = bioParameters::the()->getValueInt("deriveAnalyticalHessian",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  return (h != 0) ;
}

patBoolean bioPrecompiledFunction::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

unsigned long bioPrecompiledFunction::getDimension() const {
  return bioLiteralRepository::the()->getNumberOfEstimatedParameters() ;
}

patBoolean bioPrecompiledFunction::isUserBased() const {
  return patTRUE ;
}


void *computeFunctionForThread(void* fctPtr) {
  patError* err(NULL) ;
  bioThreadArg *input = (bioThreadArg *) fctPtr;
  input->err = NULL ;

  bioSample* sample = input->sample ;
  bioIteratorSpan __theThreadSpan = *input->subsample ; 
  bioExpressionRepository* theRep = input->theExpressionRepository ;
//  DEBUG_MESSAGE("Getting expr with id " << input->theExpressionId);
  bioExpression* theExpression = theRep->getExpression(input->theExpressionId) ;
  if (theExpression == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    input->err = err ;
    return NULL;
  }
//  DEBUG_MESSAGE("set sample");


  theExpression->setSample(sample) ;

//  DEBUG_MESSAGE("set thread span");
  theExpression->setThreadSpan(__theThreadSpan) ;
  
  // DEBUG_MESSAGE("THREAD " << input->threadId <<" LIST OF EXPRESSIONS") ;
  // cout << bioExpressionRepository::the()->printList() << endl ;

  input->result = theExpression->getValue(patFALSE,patLap::get(),err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    input->err = err ;
  }
  return NULL ;
}

void *computeFunctionAndGradientForThread( void *ptr ) {
  patError* err(NULL) ;
  bioThreadArg *input = (bioThreadArg *) ptr;
  input->err = NULL ;
  bioSample* sample = input->sample ;
  bioIteratorSpan __theThreadSpan = *input->subsample ; 
  vector<patULong> literalIds = input->literalIds ;
  patULong exprId = input->theExpressionId ;
  bioExpressionRepository* theRepository = input->theExpressionRepository ;
  bioExpression* theExpression = theRepository->getExpression(exprId) ;
  if (theExpression == NULL) {
    err = new patErrNullPointer("bioExpression") ;
    WARNING(err->describe()) ;
    input->err = err ;
    return NULL;
  }
  
  theExpression->setSample(sample) ;
  theExpression->setThreadSpan(__theThreadSpan) ;
  
  //bioFunctionAndDerivatives result = theExpression->getFunctionAndGradient(err) ;
  patBoolean debugDeriv(patFALSE) ;
#ifdef DEBUG
  debugDeriv = (bioParameters::the()->getValueInt("debugDerivatives",err) != 0) ;
#endif 
  if (err != NULL) {
    WARNING(err->describe()) ;
    input->err = err ;
    return NULL ;
  }
  bioFunctionAndDerivatives* result = 
    theExpression->getNumericalFunctionAndGradient(literalIds,
						   patBoolean(input->hessian != NULL),
						   debugDeriv,
						   err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    input->err = err ;
    return NULL ;
  }
  
  if (result->theGradient.size() != input->grad->size()) {
    stringstream str ;
    str << "Incompatible sizes: " << result->theGradient.size() << " <> " << input->grad->size() ;
    err = new patErrMiscError(str.str());
    WARNING(err->describe());
    input->err = err ;
    return NULL ;
  }
  
  input->result = result->theFunction ;
  for (patULong i = 0 ; i < result->theGradient.size() ; ++i) {
    (*input->grad)[i] = result->theGradient[i] ;
  }

  trHessian* bhhh = theExpression->getBhhh() ;
  if (bhhh != NULL) {
    
    input->bhhh->set(*bhhh,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      input->err = err ;
      return NULL ;
    }
  }

  if (input->hessian != NULL) {
    trHessian* theHessian = result->theHessian ;
    if (theHessian == NULL) {
      WARNING("Error: NULL pointer") ;
      return NULL ;
    }
    input->hessian->set(*theHessian,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      input->err = err ;
      return NULL ;
    }
  }
  return NULL ;
}

void bioPrecompiledFunction::generateCppCode(ostream& str, patError*& err) {
  err = new patErrMiscError("Not implemented") ;
  WARNING(err->describe())  ;
  return ;
}
