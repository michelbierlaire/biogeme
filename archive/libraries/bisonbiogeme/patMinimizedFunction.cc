//-*-c++-*------------------------------------------------------------
//
// File name : patMinimizedFunction.cc
// Author :    Michel Bierlaire
// Date :      Wed Aug  9 01:48:54 2000
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <fstream>
#include <iomanip>

#include "patConst.h"
#include "patMath.h"
#include "patArithNode.h"
#include "patModelSpec.h"
#include "patParameters.h"
#include "patMinimizedFunction.h"
#include "patLikelihood.h"
#include "patSample.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patErrOutOfRange.h"
#include "patTimer.h"
#include "patSecondDerivatives.h"

patMinimizedFunction::patMinimizedFunction(patLikelihood* aLikelihood, trParameters p) :
  likelihood(aLikelihood) , 
  firstTime(patTRUE) , 
  useBhhh(patFALSE),
  secondDeriv(NULL),
  theTrParameters(p) {

  // Compute BHHH anyway.
  useBhhh = patTRUE ;
  unsigned long n = patModelSpec::the()->getNbrNonFixedParameters() ;
  bhhh.resize(n) ;
  fill(bhhh.begin(),bhhh.end(),patVariables(n,0.0)) ;
  if (patModelSpec::the()->isSimpleMnlModel() && 
      patModelSpec::the()->isMuFixed()) {
    secondDeriv = new patSecondDerivatives(patModelSpec::the()->getNbrTotalBeta()) ;
  }
}

patMinimizedFunction::~patMinimizedFunction() {

}

patReal patMinimizedFunction::computeFunction(trVector* x,
					      patBoolean* success,
					      patError*& err) {

  if (*x == previousTime) {
    *success = patTRUE ;
    return previousValue ;
  }
  else {
    previousTime = *x ;
  }

  if (err != NULL) {
    WARNING(err->describe());
    *success = patFALSE ;
    return patReal();
  }

  if (x == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;
  }

  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal() ;
  }

  if (likelihood == NULL) {
    err = new patErrNullPointer("patLikelihood") ;
    WARNING(err->describe()) ;
    *success = patFALSE ;
    return patReal()  ;
  }

  patModelSpec::the()->setEstimatedCoefficients(x,err) ;

  if (err != NULL) {
    WARNING(err->describe());
    *success = patFALSE ;
    return patReal();
  }

  patReal res = 
    likelihood->evaluateLog(patModelSpec::the()->getPtrBetaParameters(),
			    patModelSpec::the()->getPtrMu(),
			    patModelSpec::the()->getPtrModelParameters(),
			    patModelSpec::the()->getPtrScaleParameters(),
			    patTRUE,
			    vector<patBoolean>(),
			    vector<patBoolean>(),
			    patFALSE,
			    vector<patBoolean>(),
			    NULL,
			    NULL,
			    NULL,
			    NULL,
			    NULL,
			    success,
			    NULL,
			    patFALSE,
			    NULL,
			    NULL,
			    err) ;

  if (err != NULL) {
    WARNING(err->describe());
    return patReal();
  }

  // Because we minimize, we return the opposite
  
  if (!(*success)) {
    //    DEBUG_MESSAGE("Not successfull " << patMaxReal) ;
    previousValue = patMaxReal ;
    return patMaxReal ;
  }
  previousValue = -res ;
  return (-res) ;
}

patReal patMinimizedFunction::computeFunctionAndDerivatives(trVector* x,
							    trVector* grad,
							    trHessian* trueHessian,
							    patBoolean* success,
							    patError*& err) {


  if (err != NULL) {
    WARNING(err->describe());
    return patReal();
  }


  if (!isHessianAvailable() && trueHessian != NULL) {
    stringstream str ;
    str << "Cannot use analytical hessian because " ;
    if (!patModelSpec::the()->isSimpleMnlModel()) {
      str << "the model is not logit " ;
    }
    if (!patModelSpec::the()->isMuFixed()) {
      str << "the mu parameter must be estimated " ;
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }





  if (success == NULL) {
    err = new patErrNullPointer("patBoolean") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (likelihood == NULL) {
    err = new patErrNullPointer("patLikelihood") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }

  if (grad == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return patReal()  ;
  }


  if (grad->size() != x->size()) {
    stringstream str ;
    str << "Incompatible sizes between x (" << x->size() 
	<< ") and grad (" << grad->size() << ")" ;
    err = new patErrMiscError(str.str()) ;
    return patReal() ;
  }

  patModelSpec::the()->setEstimatedCoefficients(x,err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return patReal();
  }

  // This function will be actually executed only once.
  if (firstTime) {
    allocateMemory(err) ;
    firstTime = patFALSE ;
  }

  for (unsigned long i = 0 ; i < betaDerivatives.size() ; ++i) {
    betaDerivatives[i] = 0.0 ;
  }
  for (unsigned long i = 0 ; i < paramDerivative.size() ; ++i) {
    paramDerivative[i] = 0.0 ;
  }
  for (unsigned long i = 0 ; i < scaleDerivative.size() ; ++i) {
    scaleDerivative[i] = 0.0 ;
  }

  for (unsigned long i = 0 ; i < grad->size() ; ++i) {
    (*grad)[i] = 0.0 ;
  }
  muDerivative = 0.0 ;

  fill(bhhh.begin(),bhhh.end(),patVariables(grad->size(),0.0)) ;
  if (secondDeriv != NULL) {
    if (trueHessian != NULL) {
      trueHessian->setToZero() ;
    }
    secondDeriv->setToZero() ;
  }


  patReal res = 
    likelihood->evaluateLog(patModelSpec::the()->getPtrBetaParameters(),
			    patModelSpec::the()->getPtrMu(),
			    patModelSpec::the()->getPtrModelParameters(),
			    patModelSpec::the()->getPtrScaleParameters(),
			    patFALSE,
			    betaVariable,
			    paramVariable,
			    muVariable,
			    scaleVariable,
			    &betaDerivatives,
			    &paramDerivative,
			    &muDerivative,
			    secondDeriv,
			    &scaleDerivative,
			    success,
			    grad,
			    useBhhh,
			    &bhhh,
			    trueHessian,
			    err) ;

  if (err != NULL) {
    WARNING(err->describe());
    return patReal();
  }


//   patModelSpec::the()->gatherGradient(grad,
// 				      betaDerivatives,
// 				      scaleDerivative,
// 				      paramDerivative,
// 				      muDerivative,
// 				      err) ;

//   if (err != NULL) {
//     WARNING(err->describe());
//     return patReal();
//   }
  
  // Because we minimize, we return the opposite
  
  if (!(*success)) {
    DEBUG_MESSAGE("Not successfull " << patMaxReal) ;
    return patMaxReal ;
  }

   // DEBUG_MESSAGE("BHHH = ") ;
   // for (vector<patVariables>::iterator ii = bhhh.begin() ;
   //      ii != bhhh.end() ;
   //      ++ii) {
   //   DEBUG_MESSAGE(*ii) ;
   // }


  return (-res) ;
}

trVector* patMinimizedFunction::computeHessianTimesVector(trVector* x,
						     const trVector* v,
							  trVector* r,
						     patBoolean* success,
						     patError*& err)  {
 
  err = new patErrMiscError("Not implemented") ;
  WARNING(err->describe()) ;
  return NULL ;
}

patBoolean patMinimizedFunction::isGradientAvailable() const {
  return patTRUE ;
}

patBoolean patMinimizedFunction::isHessianAvailable() const {

  if (!patModelSpec::the()->isSimpleMnlModel()) {
    return patFALSE ;
  }
  if (!patModelSpec::the()->isMuFixed()) {
    return patFALSE ;
  }
  return patTRUE ;

}

patBoolean patMinimizedFunction::isHessianTimesVectorAvailable() const {
  return patFALSE ;
}

unsigned long patMinimizedFunction::getDimension() const {
  return patModelSpec::the()->getNbrNonFixedParameters() ;
}





void patMinimizedFunction::allocateMemory(patError*& err) {
  
//    DEBUG_MESSAGE("Allocating memory for ") ;
//    DEBUG_MESSAGE('\t' << patModelSpec::the()->getNbrTotalBeta() << " beta's") ;
//    DEBUG_MESSAGE('\t' << patModelSpec::the()->getNbrModelParameters() << " model parameters") ;
//    DEBUG_MESSAGE('\t' << patModelSpec::the()->getNbrScaleParameters() << " scale parameters") ;

  if (err != NULL) {
    WARNING(err->describe());
    return;
  }

  betaVariable.resize(patModelSpec::the()->getNbrTotalBeta()) ;
  betaDerivatives.resize(patModelSpec::the()->getNbrTotalBeta()) ;
  paramVariable.resize(patModelSpec::the()->getNbrModelParameters()) ;
  paramDerivative.resize(patModelSpec::the()->getNbrModelParameters()) ;
  scaleVariable.resize(patModelSpec::the()->getNbrScaleParameters()) ;
  scaleDerivative.resize(patModelSpec::the()->getNbrScaleParameters()) ;

  patIterator<patBetaLikeParameter>* betaIter = 
    patModelSpec::the()->createAllBetaIterator() ;
  if (betaIter == NULL) {
    err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
    WARNING(err->describe()) ;
    return  ;
  }
  
  for (betaIter->first() ;
       !betaIter->isDone() ;
       betaIter->next()) {
    patBetaLikeParameter bb = betaIter->currentItem() ;
    if (bb.id >= betaVariable.size()) {
	err = new patErrOutOfRange<unsigned long>(bb.id,
						   0,
						   betaVariable.size()-1) ;
	WARNING(err->describe()) ;
	return  ;
    }
    betaVariable[bb.id] = !bb.isFixed ;
  }


  patIterator<patBetaLikeParameter>* modelIter = 
    patModelSpec::the()->createAllModelIterator() ;
  if (modelIter == NULL) {
    err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
    WARNING(err->describe()) ;
    return  ;
  }
  
  for (modelIter->first() ;
       !modelIter->isDone() ;
       modelIter->next()) {
    patBetaLikeParameter bb = modelIter->currentItem() ;
    if (bb.id >= paramVariable.size()) {
      err = new patErrOutOfRange<unsigned long>(bb.id,
						 0,
						 paramVariable.size()-1) ;
      WARNING(err->describe()) ;
      return  ;
    }
    paramVariable[bb.id] = !bb.isFixed ;
  }

  patIterator<patBetaLikeParameter>* scaleIter = 
    patModelSpec::the()->createScaleIterator() ;
  if (scaleIter == NULL) {
    err = new patErrNullPointer("patIterator<patBetaLikeParameter>") ;
    WARNING(err->describe()) ;
    return  ;
  }
  
  for (scaleIter->first() ;
       !scaleIter->isDone() ;
       scaleIter->next()) {
    patBetaLikeParameter bb = scaleIter->currentItem() ;
    if (bb.id >= scaleVariable.size()) {
      err = new patErrOutOfRange<unsigned long>(bb.index,
						 0,
						 scaleVariable.size()-1) ;
      WARNING(err->describe()) ;
      return  ;
    }
    scaleVariable[bb.id] = !bb.isFixed ;
  }

  patBetaLikeParameter mu = patModelSpec::the()->getMu(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  muVariable = !mu.isFixed ;

  
}

trHessian* patMinimizedFunction::computeCheapHessian(trHessian* theCheapHessian,
						 patError*& err) {
  if (theCheapHessian == NULL) {
    err = new patErrNullPointer("trHessian") ;
    WARNING(err->describe()) ;
    return NULL ;
  }

  patULong n = theCheapHessian->getDimension() ;
  if (n != getDimension()) {
    stringstream str ;
    str << "Incompatibel sizes " << n << " and " << getDimension();
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (useBhhh) {
    for (unsigned long i = 0 ; i < n ; ++i) {
      for (unsigned long j = i ; j < n ; ++j) {
	theCheapHessian->setElement(i,j,bhhh[i][j],err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL ;
	}
      }
    }
  }
  else {
    for (unsigned long i = 0 ; i < n ; ++i) {
      for (unsigned long j = i ; j < n ; ++j) {
	if ( i != j) {
	  theCheapHessian->setElement(i,j,0.0,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL ;
	  }
	}
      }
      theCheapHessian->setElement(i,i,1.0,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
  }
  //  DEBUG_MESSAGE(hessian) ;
  return theCheapHessian ;
}
patBoolean patMinimizedFunction::isCheapHessianAvailable() {
  return useBhhh ;
}

void patMinimizedFunction::generateCppCode(ostream& cppFile, patError*& err) {
  
  patBoolean isPanel = patModelSpec::the()->isPanelData() ;
  cppFile << "////////////////////////////////////////" << endl ;
  cppFile << "// Code generated in patMinimizedFunction" << endl ;

  cppFile << "#include <pthread.h> //POSIX Threads library" << endl ;
  cppFile << "#define NUMTHREADS " << patParameters::the()->getgevNumberOfThreads() << endl ;
  cppFile << "#include \"patBiogemeScripting.h\"" << endl ;
  cppFile << "#include \"patFastBiogeme.h\"" << endl ;
  cppFile << "#include \"patFileNames.h\"" << endl ;
  cppFile << "#include \"patVersion.h\"" << endl ;
  cppFile << "#include \"patSample.h\"" << endl ;
  cppFile << "#include \"patErrNullPointer.h\"" << endl ;
  if (!isPanel) {
    cppFile << "#include \"patObservationData.h\"" << endl ;
  }
  else {
    cppFile << "#include \"patIndividualData.h\"" << endl ;
  }
  cppFile << "#include \"patPower.h\"" << endl ;

  cppFile << "" << endl ;

  cppFile << "// Data structure for the threads" << endl ;
  cppFile << "typedef struct{" << endl ;
  if (!isPanel) {
    cppFile << "  patIterator<patObservationData*>* theObsIterator ;" << endl ;
  }
  else {
    cppFile << "  patIterator<patIndividualData*>* theIndIterator ;" << endl ;
  }
  cppFile << "  trVector* x ;" << endl ;
  cppFile << "  trVector* grad;" << endl ;
  cppFile << "  trHessian* bhhh ;" << endl ;
  // if (trueHessian) {
  //   cppFile << "  trHessian* trueHessian ;" << endl ;
  // }
  cppFile << "  patBoolean* success ;" << endl ;
  cppFile << "  patError* err ;" << endl ;
  cppFile << "  patReal result ;" << endl ;
  cppFile << "} inputArg ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "void *computeFunction( void *ptr );" << endl ;
  cppFile << "void *computeFunctionAndGradient( void *ptr );" << endl ;
  cppFile << "" << endl ;
  cppFile << "// Definition of the class" << endl ;
  cppFile << "" << endl ;
  cppFile << "class myBiogeme: public patFastBiogeme {" << endl ;
  cppFile << "" << endl ;
  cppFile << "public :" << endl ;
  cppFile << "" << endl ;
  cppFile << "  myBiogeme() ;" << endl ;
  cppFile << "  ~myBiogeme() {};" << endl ;
  cppFile << "  patReal getFunction(trVector* x," << endl ;
  cppFile << "		      patBoolean* success," << endl ;
  cppFile << "		      patError*& err) ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  trVector* getGradient(trVector* x," << endl ;
  cppFile << "			trVector* grad," << endl ;
  cppFile << "			patBoolean* success," << endl ;
  cppFile << "			patError*& err)  ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  patReal getFunctionAndGradient(trVector* x," << endl ;
  cppFile << "				 trVector* grad," << endl ;
  cppFile << "				 patBoolean* success," << endl ;
  cppFile << "				 patError*& err); " << endl ;
  cppFile << "" << endl ;
  cppFile << "  trHessian* computeHessian(patVariables* x," << endl ;
  cppFile << "			    trHessian& hessian," << endl ;
  cppFile << "			    patBoolean* success," << endl ;
  cppFile << "			    patError*& err) ; " << endl ;
  cppFile << "  " << endl ;
  cppFile << "  trHessian* getCheapHessian(trVector* x," << endl ;
  cppFile << "			     trHessian& hessian," << endl ;
  cppFile << "			     patBoolean* success," << endl ;
  cppFile << "			     patError*& err) ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "   patBoolean isCheapHessianAvailable() ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  trVector getHessianTimesVector(trVector* x," << endl ;
  cppFile << "				 const trVector* v," << endl ;
  cppFile << "				 patBoolean* success," << endl ;
  cppFile << "				 patError*& err)  ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  patBoolean isGradientAvailable(patError*& err) const ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patBoolean isHessianAvailable(patError*& err)  const ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patBoolean isHessianTimesVectorAvailable(patError*& err) const ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  unsigned long getDimension(patError*& err) const ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  trVector getCurrentVariables(patError*& err) const ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  patBoolean isUserBased() const ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  void generateCppCode(patError*& err) ;" << endl ;
  cppFile << "" << endl ;
  cppFile << " private:" << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << "  trHessian trueHessian ;" << endl ;
  // }
  cppFile << "  trHessian bhhh ;" << endl ;

  if (!isPanel) {
    cppFile << "  patObservationData* observation ;" << endl ;
    cppFile << "  patAggregateObservationData* aggObservation ;" << endl ;
    cppFile << "  patIterator<patAggregateObservationData*>* theAggObsIterator ;" << endl ;
  }
  else {
    cppFile << "  patIndividualData* individual ;" << endl ;
  }
  if (patModelSpec::the()->estimateGroupScales()) {
    cppFile << "  map<long, unsigned long> groupIndex ;" << endl ;
  }
  cppFile << "  vector<unsigned long> altIndex ;" 
	  << endl ;
  cppFile << "  vector<trVector> threadGrad ;" << endl ;
  cppFile << "  vector<trHessian> threadBhhh ;" << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << "  vector<trHessian> threadHessian ;" << endl ;
  // }

  cppFile << "};" << endl ;
  
  
  cppFile << "" << endl ;
  cppFile << "// Implementation of the class" << endl ;
  cppFile << "" << endl ;
  
  cppFile << "  myBiogeme::myBiogeme() : altIndex("
	  << 1+patModelSpec::the()->getLargestAlternativeUserId() 
	  <<") " << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << ", trueHessian("<< getDimension() << ")" << endl ;
  // }
  cppFile << ", bhhh("<< getDimension() << ")" << endl ;
  cppFile << " , threadGrad(NUMTHREADS,trVector("<< getDimension() <<"))" << endl ;
  cppFile << " , threadBhhh(NUMTHREADS,trHessian("<<getDimension()<<"))" << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << " , threadHessian(NUMTHREADS,trHessian("<<getDimension()<<"))" ;
  // }
  cppFile << "    { " << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << "  trueHessian.setToZero() ;" << endl ;
  // }
    cppFile << "  bhhh.setToZero() ;" << endl ;
    //  patModelSpec::the()->generateCppCodeForAltId(cppFile,err) ;
//   if (patModelSpec::the()->estimateGroupScales()) {
//     sample->generateCppCodeForGroupId(cppFile,err) ;
//     if (err != NULL) {
//       WARNING(err->describe()) ;
//     }
//   }
  cppFile << "    };" << endl ;
  cppFile << "" << endl ;
  cppFile << "patReal myBiogeme::getFunction(trVector* x," << endl ;
  cppFile << "			       patBoolean* success," << endl ;
  cppFile << "			       patError*& err) {" << endl ;
  cppFile << "" << endl ;

  cppFile << "  patReal logLike = 0.0 ;" << endl ;

  if (!isPanel) {
    cppFile << "  vector<patIterator<patObservationData*>*> theObsIterator = " << endl ;
    cppFile << "    sample->createObsIteratorThread(NUMTHREADS,err) ;" << endl ;
    cppFile << "  if (err != NULL) {" << endl ;
    cppFile << "    WARNING(err->describe()) ;" << endl ;
    cppFile << "    return NULL ;" << endl ;
    cppFile << "  }" << endl ;
  }
  else {
    cppFile << "  vector<patIterator<patIndividualData*>*> theIndIterator = " << endl ;
    cppFile << "    sample->createIndIteratorThread(NUMTHREADS,err) ;" << endl ;
    cppFile << "  if (err != NULL) {" << endl ;
    cppFile << "    WARNING(err->describe()) ;" << endl ;
    cppFile << "    return NULL ;" << endl ;
    cppFile << "  }" << endl ;

  }
      cppFile << "  pthread_t *threads;" << endl ;
      cppFile << "  threads = new pthread_t[NUMTHREADS];" << endl ;
      cppFile << "" << endl ;
      cppFile << "  inputArg *input;" << endl ;
      cppFile << "  input = new inputArg[NUMTHREADS];" << endl ;
      cppFile << "" << endl ;
      cppFile << "  for(int i=0; i<NUMTHREADS; i++) {" << endl ;
      if (!isPanel) {
	cppFile << "    input[i].theObsIterator = theObsIterator[i] ;" << endl ;
      }
      else {
	cppFile << "    input[i].theIndIterator = theIndIterator[i] ;" << endl ;

      }
      cppFile << "    input[i].x = x ;" << endl ;
      cppFile << "    input[i].grad = NULL ;" << endl ;
      // if (trueHessian) {
      // 	cppFile << "    input[i].trueHessian = NULL ;" << endl ;
      // }
      cppFile << "    input[i].bhhh = NULL ;" << endl ;
      cppFile << "    input[i].success = success ;" << endl ;
      cppFile << "    input[i].err = err ;" << endl ;
      cppFile << "    input[i].result = 0.0 ;" << endl ;

      cppFile << "    if(pthread_create( &threads[i], NULL, computeFunction, (void*) &input[i]))" << endl ;
      cppFile << "      {" << endl ;
      cppFile << "	err = new patErrMiscError(\"Error creating Threads!\") ;" << endl ;
      cppFile << "	WARNING(err->describe()) ;" << endl ;
      cppFile << "	return NULL ;" << endl ;
      cppFile << "      }" << endl ;
      cppFile << "  }" << endl ;
      cppFile << "  " << endl ;
      cppFile << "  void *tmp;" << endl ;
      cppFile << "  " << endl ;
      cppFile << "  patBoolean maxreal(patFALSE) ;" << endl ;
      cppFile << "  for(int i=0; i<NUMTHREADS; ++i) {" << endl ;
      cppFile << "    " << endl ;
      cppFile << "    pthread_join( threads[i], &tmp);" << endl ;
      cppFile << "" << endl ;
      cppFile << "      if (!maxreal) {" << endl ;
      cppFile << "	if (input[i].result == -patMaxReal) {" << endl ;
      cppFile << "	  logLike = patMaxReal ;" << endl ;
      cppFile << "	  maxreal = patTRUE ;" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "	else {" << endl ;
      cppFile << "    logLike -= input[i].result ;" << endl ;
      cppFile << "	}" << endl ;
      cppFile << "      }" << endl ;
      cppFile << "" << endl ;
      cppFile << "  }" << endl ;


  cppFile << "  // Because we minimize, we return the opposite" << endl ;
  cppFile << "  return (logLike) ;" << endl ;


  cppFile << "}" << endl ;
  cppFile << "  " << endl ;
  cppFile << "trVector* myBiogeme::getGradient(trVector* x," << endl ;
  cppFile << "				 trVector* grad," << endl ;
  cppFile << "				 patBoolean* success," << endl ;
  cppFile << "				 patError*& err) {" << endl ;
  
  cppFile << "  patReal result = getFunctionAndGradient(x,grad,success,err) ;" << endl ;
  cppFile << "  if (err != NULL) {" << endl ;
  cppFile << "    WARNING(err->describe()) ;" << endl ;
  cppFile << "    return NULL  ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  return grad ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "  " << endl ;
  cppFile << "patReal myBiogeme::getFunctionAndGradient(trVector* x," << endl ;
  cppFile << "					  trVector* grad," << endl ;
  cppFile << "					  patBoolean* success," << endl ;
  cppFile << "					  patError*& err) {" << endl ;


  cppFile << "  patReal logLike = 0.0 ;" << endl ;
  cppFile << "  fill(grad->begin(),grad->end(),0.0) ;" << endl ;
  cppFile << "  bhhh.setToZero() ;" ;
  // if (trueHessian != NULL) {
  //   cppFile << "        trueHessian.setToZero() ;" ;


  // }
  cppFile << "  " << endl ;
  if (!isPanel) {
    cppFile << "  vector<patIterator<patObservationData*>*> theObsIterator = " << endl ;
    cppFile << "    sample->createObsIteratorThread(NUMTHREADS,err) ;" << endl ;
    cppFile << "  if (err != NULL) {" << endl ;
    cppFile << "    WARNING(err->describe()) ;" << endl ;
    cppFile << "    return patReal() ;" << endl ;
    cppFile << "  }" << endl ;
    cppFile << "" << endl ;
  }
  else {
    cppFile << "  vector<patIterator<patIndividualData*>*> theIndIterator = " << endl ;
    cppFile << "    sample->createIndIteratorThread(NUMTHREADS,err) ;" << endl ;
    cppFile << "  if (err != NULL) {" << endl ;
    cppFile << "    WARNING(err->describe()) ;" << endl ;
    cppFile << "    return patReal() ;" << endl ;
    cppFile << "  }" << endl ;
    cppFile << "" << endl ;

  }
  cppFile << "  pthread_t *threads;" << endl ;
  cppFile << "  threads = new pthread_t[NUMTHREADS];" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  inputArg *input;" << endl ;
  cppFile << "  input = new inputArg[NUMTHREADS];" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  for(int i=0; i<NUMTHREADS; i++) {" << endl ;
  if (!isPanel) {
    cppFile << "    input[i].theObsIterator = theObsIterator[i] ;" << endl ;
  }
  else {
    cppFile << "    input[i].theIndIterator = theIndIterator[i] ;" << endl ;
  }
  cppFile << "    input[i].x = x ;" << endl ;
  cppFile << "    fill(threadGrad[i].begin(),threadGrad[i].end(),0.0) ;" << endl  ;
  cppFile << "    input[i].grad = &(threadGrad[i]) ;" << endl ;
  cppFile << "    threadBhhh[i].setToZero() ;" ;
  cppFile << "    input[i].bhhh = &(threadBhhh[i]) ;" << endl ;
  // if (trueHessian != NULL) {
  //   cppFile << "    threadHessian[i].setToZero() ;" << endl ;
  //   cppFile << "    input[i].trueHessian = &(threadHessian[i]) ;" << endl ;
  // }
  cppFile << "    input[i].success = success ;" << endl ;
  cppFile << "    input[i].err = err ;" << endl ;
  cppFile << "    input[i].result = 0.0 ;" << endl ;
  cppFile << "    if(pthread_create( &threads[i], NULL, computeFunctionAndGradient, (void*) &input[i])) {" << endl ;
  cppFile << "      err = new patErrMiscError(\"Error creating Threads! Exit!\") ;" << endl ;
  cppFile << "      WARNING(err->describe()) ;" << endl ;
  cppFile << "      return 0.0;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  void *tmp;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "  patBoolean maxreal(patFALSE) ;" << endl ;
  cppFile << "  for(int i=0; i<NUMTHREADS; ++i) {" << endl ;
  cppFile << "    pthread_join( threads[i], &tmp);" << endl ;
  cppFile << "    if (!maxreal) {" << endl ;
  cppFile << "      if (input[i].result == -patMaxReal) {" << endl ;
  cppFile << "	logLike = patMaxReal ;" << endl ;
  cppFile << "	maxreal = patTRUE ;" << endl ;
  cppFile << "      }" << endl ;
  cppFile << "      else {" << endl ;
  cppFile << "    logLike -= input[i].result ;" << endl ;
  cppFile << "    (*grad) -= threadGrad[i] ;" << endl ;
  cppFile << "    bhhh.add(1.0,threadBhhh[i],err) ;" << endl ;
  cppFile << "    if (err != NULL) {" << endl ;
  cppFile << "      WARNING(err->describe()) ;" << endl ;
  cppFile << "      return 0.0;" << endl ;
  cppFile << "      }" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "    }" << endl ;
  // if (trueHessian) {
  //   cppFile << "    trueHessian.add(1.0,threadHessian[i],err) ;" << endl ;
  // }
  cppFile << "    if (err != NULL) {" << endl ;
  cppFile << "      WARNING(err->describe()) ;" << endl ;
  cppFile << "      return 0.0;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "	" << endl ;
  cppFile << "  return logLike ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "trHessian* myBiogeme::computeHessian(patVariables* x," << endl ;
  cppFile << "				     trHessian& hessian," << endl ;
  cppFile << "				     patBoolean* success," << endl ;
  cppFile << "				     patError*& err) {" << endl ;
  // if (secondDeriv != NULL && trueHessian != NULL) {
  //     cppFile << "    hessian = trueHessian ;" << endl ;
  //     cppFile << "    *success = patTRUE ;" << endl ;
  //     cppFile << "    return &trueHessian ;" << endl ;
  // }
  // else {
    cppFile << "    return computeFinDiffHessian(x,hessian,success,err) ;" << endl ;
  // }
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "trHessian* myBiogeme::getCheapHessian(trVector* x," << endl ;
  cppFile << "			     trHessian& hessian," << endl ;
  cppFile << "			     patBoolean* success," << endl ;
  cppFile << "				      patError*& err) {" << endl ;
  cppFile << "    hessian = bhhh ;" << endl ;
  cppFile << "    *success = patTRUE ;" << endl ;
  cppFile << "    return &bhhh ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patBoolean myBiogeme::isCheapHessianAvailable() {" << endl ;
  cppFile << "  return patTRUE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "trVector myBiogeme::getHessianTimesVector(trVector* x," << endl ;
  cppFile << "					  const trVector* v," << endl ;
  cppFile << "					  patBoolean* success," << endl ;
  cppFile << "					  patError*& err) {" << endl ;
  cppFile << "  return trVector() ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patBoolean myBiogeme::isGradientAvailable(patError*& err) const {" << endl ;
  cppFile << "  return patTRUE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patBoolean myBiogeme::isHessianAvailable(patError*& err)  const {" << endl ;
  cppFile << "  return patTRUE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patBoolean myBiogeme::isHessianTimesVectorAvailable(patError*& err) const {" << endl ;
  cppFile << "  return patFALSE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "unsigned long myBiogeme::getDimension(patError*& err) const {" << endl ;
  cppFile << "  return "<<patModelSpec::the()->getNbrNonFixedParameters()<<" ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "  " << endl ;
  cppFile << "patBoolean myBiogeme::isUserBased() const {" << endl ;
  cppFile << "  return patTRUE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "void myBiogeme::generateCppCode(patError*& err) {" << endl ;
  cppFile << "  err = new patErrMiscError(\"This function should never be called\") ;" << endl ;
  cppFile << "  WARNING(err->describe()) ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "// Implementation of the default display  on a terminal" << endl ;
  cppFile << "" << endl ;
  cppFile << "patDisplay::patDisplay() : screenImportance(patImportance::patDEBUG), " << endl ;
  cppFile << "			   logImportance(patImportance::patDEBUG)," << endl ;
  cppFile << "			   logFile(patFileNames::the()->getLogFile().c_str()) {" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patAbsTime now ;" << endl ;
  cppFile << "  now.setTimeOfDay() ;" << endl ;
  cppFile << "  logFile << \"This file has automatically been generated.\" << endl ;" << endl ;
  cppFile << "  logFile << now.getTimeString(patTsfFULL) << endl ;" << endl ;
  cppFile << "  logFile << patVersion::the()->getCopyright() << endl ;" << endl ;
  cppFile << "  logFile << endl ;" << endl ;
  cppFile << "  logFile << patVersion::the()->getVersionInfoDate() << endl ;" << endl ;
  cppFile << "  logFile << patVersion::the()->getVersionInfoAuthor() << endl ;" << endl ;
  cppFile << "  logFile << endl ;" << endl ;
  cppFile << "  logFile << flush ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patDisplay::~patDisplay() {" << endl ;
  cppFile << "  messages.erase(messages.begin(),messages.end()) ;" << endl ;
  cppFile << "   logFile.close() ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patDisplay* patDisplay::the() {" << endl ;
  cppFile << "  static patDisplay* singleInstance = NULL ;" << endl ;
  cppFile << "  if (singleInstance == NULL) {" << endl ;
  cppFile << "    singleInstance = new patDisplay ;" << endl ;
  cppFile << "    assert(singleInstance != NULL) ;" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  return singleInstance ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "void patDisplay::addMessage(const patImportance& aType," << endl ;
  cppFile << "			    const patString& text," << endl ;
  cppFile << "			    const patString& fileName," << endl ;
  cppFile << "			    const patString& lineNumber) {" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patMessage theMessage ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  theMessage.theImportance = aType ;" << endl ;
  cppFile << "  theMessage.text = text ;" << endl ;
  cppFile << "  theMessage.fileName = fileName ;" << endl ;
  cppFile << "  theMessage.lineNumber = lineNumber ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patAbsTime now ;" << endl ;
  cppFile << "  now.setTimeOfDay() ;" << endl ;
  cppFile << "  theMessage.theTime = now ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  if (aType <= screenImportance) {" << endl ;
  cppFile << "    if (screenImportance < patImportance::patDEBUG) {" << endl ;
  cppFile << "      cout << theMessage.shortVersion() << endl << flush  ;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "    else {" << endl ;
  cppFile << "      cout << theMessage.fullVersion() << endl << flush  ;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  if (aType <= logImportance) {" << endl ;
  cppFile << "    if (logImportance < patImportance::patDEBUG) {" << endl ;
  cppFile << "      logFile << theMessage.shortVersion() << endl << flush  ;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "    else {" << endl ;
  cppFile << "      logFile << theMessage.fullVersion() << endl << flush  ;" << endl ;
  cppFile << "    }" << endl ;
  cppFile << "  }" << endl ;
  cppFile << "  messages.push_back(theMessage) ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "void patDisplay::setScreenImportanceLevel(const patImportance& aType) {" << endl ;
  cppFile << "  screenImportance = aType ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "  " << endl ;
  cppFile << "void patDisplay::setLogImportanceLevel(const patImportance& aType) {" << endl ;
  cppFile << "  logImportance = aType ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "void patDisplay::initProgressReport(const patString message," << endl ;
  cppFile << "			unsigned long upperBound) {" << endl ;
  cppFile << "  " << endl ;
  cppFile << "" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "patBoolean patDisplay::updateProgressReport(unsigned long currentValue) {" << endl ;
  cppFile << "  return patTRUE ;" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "void patDisplay::terminateProgressReport() {" << endl ;
  cppFile << "" << endl ;
  cppFile << "}" << endl ;
  cppFile << "" << endl ;
  cppFile << "" << endl ;
  cppFile << "// Main program. " << endl ;
  cppFile << "// It sends the pointer to the new object to the DLL and run biogeme" << endl ;
  cppFile << "" << endl ;
  cppFile << "" << endl ;
  cppFile << "int main(int argc, char *argv[]) {" << endl ;
  cppFile << "" << endl ;
  cppFile << "  patBiogemeScripting theMain ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  myBiogeme theFunction ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  theMain.fastBiogeme(&theFunction) ;" << endl ;
  cppFile << "" << endl ;
  cppFile << "  theMain.estimate(argc,argv) ;" << endl ;
  cppFile << "  " << endl ;
  cppFile << "" << endl ;
  cppFile << "}" << endl ;


  likelihood->generateCppCode(cppFile, err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  cppFile << "// End of code generated in patMinimizedFunction" << endl ;
  cppFile << "////////////////////////////////////////" << endl ;
  /// Stop HERE
  
}
