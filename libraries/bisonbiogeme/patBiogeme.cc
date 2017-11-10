//-*-c++-*------------------------------------------------------------
//
// File name : patBiogeme.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sun Apr 10 14:29:47 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
// #include "patDonlp2.h"
 #include "patString.h"
#include "patMath.h"
 #include <list>
 #include <fstream>
 #include <numeric>
 #include <iomanip>
 #include "patDrawsFromFile.h"
 #include "patLinearUtility.h"
 #include "patErrMiscError.h"
 #include "patErrOutOfRange.h"
 #include "patOutputFiles.h"
 #include "patErrNullPointer.h"
 #include "patParameters.h"
 #include "patFileNames.h"
 #include "trNonLinearAlgo.h"
 #include "patSample.h"
 #include "patModelSpec.h"
// #include "patMNL.h"
 #include "patCNL.h"
 #include "patNL.h"
 #include "patProbaMnlModel.h"
 #include "patProbaProbitModel.h"
 #include "patProbaOrdinalLogit.h"
 #include "patProbaMnlPanelModel.h"
 #include "patProbaGevModel.h"
 #include "patProbaGevPanelModel.h"
 #include "patLikelihood.h"
 #include "patMinimizedFunction.h"
 #include "patSimBasedMaxLikeOptimization.h"
 #include "trSimBasedSimpleBoundsAlgo.h"
 #include "trSimpleBoundsAlgo.h"
 #include "trLineSearchAlgo.h"
 #include "patFileExists.h" 
 #include "patAbsTime.h"
 #include "patTimeInterval.h"
 #include "patUtility.h"
 #include "patMaxLikeProblem.h"
 #include "patCfsqp.h"
 #include "patSolvOpt.h"
// #include "trBasicTrustRegionAlgo.h"
 #include "patSampleEnumeration.h"
 #include "patErrNullPointer.h"
 #include "patVersion.h"
 #include "patFileNames.h"
 #include "patNormalWichura.h"
 #include "patCenteredUniform.h"
 #include "patTimer.h"
 #include "patSampleEnumeration.h"
#include "patSampleEnuGetIndices.h"
#include "patZhengFosgerau.h"
#include "patBiogemeIterationBackup.h"

 #include "patHalton.h" 
 #include "patHessTrain.h" 
#include "patFastBiogeme.h"
 #include "patBiogeme.h"

#ifdef GIANLUCA
#include "giaLikelihood.h"
#include "giaDuration.h"
#include "giaFixationModel.h"
#endif

patBiogeme::patBiogeme(): 
  typeOfRun(patNormalRun),
  theFastFunction(NULL),
  theNormalGenerator(NULL), 
  theRectangularGenerator(NULL),
  rng(NULL),
  gevModel(NULL),
  theSample(NULL),
  util(NULL),
  theModel(NULL),
  thePanelModel(NULL) ,
  like(NULL),
  functionToMinimize(NULL),
  dim(0),
  bounds(NULL),
  algo(NULL),
  theProblem(NULL),
  theSimBasedProblem(NULL) ,
  varCovar(NULL),
  robustVarCovar(NULL),
  dataArray(NULL),
  enuIndices(NULL)
{
}

patBiogeme::~patBiogeme() {
  //Release constraints
  for (vector<trFunction*>::iterator i = constraints.begin() ;
       i != constraints.end() ;
       ++i) {
    DELETE_PTR(*i) ;
  }

  DELETE_PTR(theNormalGenerator) ;
  DELETE_PTR(theRectangularGenerator) ;
  DELETE_PTR(rng) ;
  DELETE_PTR(gevModel) ;
  DELETE_PTR(util) ;
  DELETE_PTR(theModel) ;
  DELETE_PTR(theSample) ;
  DELETE_PTR(thePanelModel) ;
  DELETE_PTR(like) ;
  if (typeOfRun == patNormalRun) {
    DELETE_PTR(functionToMinimize) ;
  }
  DELETE_PTR(bounds) ;
  DELETE_PTR(algo);
  DELETE_PTR(theProblem) ;
  DELETE_PTR( theSimBasedProblem) ;
  DELETE_PTR(varCovar) ;
  DELETE_PTR(robustVarCovar) ;
  DELETE_PTR(enuIndices) ;
}

void patBiogeme::readParameterFile(patError*& err) {

  patString paramFile = patFileNames::the()->getParFile() ;
  if (patFileExists()(paramFile)) {  
    DEBUG_MESSAGE("Read " << paramFile) ;
    patParameters::the()->readParameterFile(paramFile) ;
  }
  else {
    GENERAL_MESSAGE("File " << paramFile << " does not exist. Default values will be used") ;
    patParameters::the()->generateMinimumParameterFile(paramFile) ;
    GENERAL_MESSAGE("A file " << paramFile << " has been created") ;
  }

  if (patFileExists()(patParameters::the()->getgevStopFileName())) {
    stringstream str ;
    str << "Please remove the file " << patParameters::the()->getgevStopFileName() << " or change the parameter gevStopFileName" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

}
void patBiogeme::checkSampleFiles(patError*& err) {

  unsigned short nbrSampleFiles = patFileNames::the()->getNbrSampleFiles() ;

  vector<long> bufferSize(nbrSampleFiles) ;
  list<patString> missingFiles ;
  for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {
    patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
    if (!patFileExists()(fileName)) {
      missingFiles.push_back(fileName) ;
    }
    else {
      ifstream f(fileName.c_str()) ;
      long size = 0 ;
      char c ;
      patBoolean stop = patFALSE ;
      while (f && !stop) {
	f.get(c) ;
	if (c == '\n') {
	  bufferSize[fileId] = size ;
	  stop = patTRUE ;
	}
	else {
	  ++size ;
	}
      }
      if (!stop) {
	stringstream str ;
	str << "Error in reading " << fileName ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  if (!missingFiles.empty()){
    stringstream str ;
    if (missingFiles.size() == 1) {
      str << "File " << *missingFiles.begin() << " is missing" << endl ;
    }
    else {
      str << "The following files are missing: " ;
      for (list<patString>::iterator i = missingFiles.begin() ;
	   i != missingFiles.end() ;
	   ++i) {
	str << "[" << *i << "]" ;
      }
    }
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  patBoolean bufferProblem = patFALSE ;
  stringstream str ;
  for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {
    patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
    if (bufferSize[fileId] >= patParameters::the()->getgevBufferSize()) {
      bufferProblem = patTRUE ;
      str << "First line of " << fileName << " contains " << bufferSize[fileId] << " chars." << endl ;
      str << "     Increase the value of parameters gevBufferSize in " 
	  << patFileNames::the()->getParFile() << " (currently " << patParameters::the()->getgevBufferSize() <<")" <<endl ;
    }
  }
  if (bufferProblem) {
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
}


void patBiogeme::initLogFiles(patError*& err) {
   debugSpecFile = patFileNames::the()->getSpecDebug() ;


  switch (patParameters::the()->getgevScreenPrintLevel()) {
  case 1:
    patDisplay::the().setScreenImportanceLevel(patImportance::patGENERAL) ;
    break ;
  case 2:
    patDisplay::the().setScreenImportanceLevel(patImportance::patDETAILED) ;
    break ;
  case 3:
    patDisplay::the().setScreenImportanceLevel(patImportance::patDEBUG) ;
    break ;
  }

  switch (patParameters::the()->getgevLogFilePrintLevel()) {
  case 1:
    patDisplay::the().setLogImportanceLevel(patImportance::patGENERAL) ;
    break ;
  case 2:
    patDisplay::the().setLogImportanceLevel(patImportance::patDETAILED) ;
    break ;
  case 3:
    patDisplay::the().setLogImportanceLevel(patImportance::patDEBUG) ;
    break ;
  }

}
void patBiogeme::readDataHeaders(patError*& err) {
  patModelSpec::the()->readDataHeader(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

// Returns patTRUE is biogeme continues after the call, patFALSE if not.
patBoolean patBiogeme::readModelSpecification(patError*& err) {


  DETAILED_MESSAGE("Read file " << patFileNames::the()->getModFile()) ;

  if (!patFileExists()(patFileNames::the()->getModFile())) {
    stringstream str ;
    str << "File " << patFileNames::the()->getModFile() << " does not exist" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patFALSE;
  }

  patModelSpec::the()->readFile(patFileNames::the()->getModFile(),err) ; 
  if (err != NULL) {
    DEBUG_MESSAGE(err->describe()) ;
    return patFALSE;
  }

  patModelSpec::the()->writeSpecFile(debugSpecFile,err) ;
  if (err != NULL) {
    DEBUG_MESSAGE(err->describe()) ;
    return patFALSE;
  }

  if (patModelSpec::the()->isSimpleMnlModel() && 
      patParameters::the()->getBTRForceExactHessianIfMnl()) {
    // The model is a simple MNL. We force the use of the exact hessian.
    patParameters::the()->setBTRExactHessian(1) ;
  }

  if (patParameters::the()->getgevGeneratePythonFile() != 0 &&
      patParameters::the()->getgevPythonFileWithEstimatedParam() == 0) {
    patString pythonSpecFile = patFileNames::the()->getPythonSpecFile(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    
    patModelSpec::the()->writePythonSpecFile(pythonSpecFile,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    GENERAL_MESSAGE("The file " << pythonSpecFile << " has been generated. Make sure to edit it before running pythonbiogeme.") ;
    return patFALSE ;
  }

  return patTRUE ;

}

void patBiogeme::initRandomNumberGenerators(patError*& err) {

  if (patParameters::the()->getgevReadDrawsFromFile()) {
    patString fn = patParameters::the()->getgevNormalDrawsFile() ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    theNormalGenerator = new patDrawsFromFile(fn,err) ;  
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    patString fu = patParameters::the()->getgevRectangularDrawsFile() ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    theRectangularGenerator = new patDrawsFromFile(fu,err) ;  
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ; 
    }
  }
  else {
    if (patParameters::the()->getgevRandomDistrib() == "HALTON") {
      result.halton = patTRUE ;
      result.hessTrain = patFALSE ;
      unsigned long nSeries = 
	patModelSpec::the()->getNbrDrawAttributesPerObservation() ;

      DETAILED_MESSAGE("Prepare Halton draws for " << nSeries << " random parameters") ;

      rng = new patHalton(nSeries,
			  patParameters::the()->getgevMaxPrimeNumber(),
			  patModelSpec::the()->getNumberOfDraws(),
			  err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else if (patParameters::the()->getgevRandomDistrib() == "MLHS") {
      result.halton = patFALSE ;
      result.hessTrain = patTRUE ;
      patUnixUniform* arng = new patUnixUniform(patParameters::the()->getgevSeed()) ;
      rng = new patHessTrain(patModelSpec::the()->getNumberOfDraws(),arng) ;
    }
    else if (patParameters::the()->getgevRandomDistrib() == "PSEUDO") {
      result.halton = patFALSE ;
      result.hessTrain = patFALSE ;
      rng = new patUnixUniform(patParameters::the()->getgevSeed()) ;
    }
    else {
      WARNING("Unknown value of the parameter gevRandomDistrib") ;
      WARNING("Valid entries are \"HALTON\", \"PSEUDO\" and \"MLHS\"") ;
      return ;
    }
    
    patNormalWichura* theNormal = new patNormalWichura() ;
    theNormal->setUniform(rng) ;
    theNormalGenerator = theNormal ;

    patCenteredUniform* theRect = new patCenteredUniform(patParameters::the()->getgevDumpDrawsOnFile()) ;
    theRect->setUniform(rng) ;
    theRectangularGenerator = theRect ;
  }
}

void patBiogeme::readSummaryParameters(patError*& err) {

  patModelSpec::the()->readSummaryParameters(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

void patBiogeme::defineGevModel(patError*& err) {

  DEBUG_MESSAGE("Model type=" << patModelSpec::the()->modelTypeName()) ;
  DEBUG_MESSAGE("Is is a simple MNL model: " << (patModelSpec::the()->isSimpleMnlModel())?" YES":" NO") ;
  if (patModelSpec::the()->isGEV()) {
    if (patModelSpec::the()->isMNL()) {
      //      gevModel = new patMNL ;
      return ;
    }
    else if (patModelSpec::the()->isCNL()) {
      gevModel = new patCNL ;
      return ;
    }
    else if (patModelSpec::the()->isNL()) {
      gevModel = new patNL(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      return ;
    }
    else if (patModelSpec::the()->isNetworkGEV()) {
      gevModel = patModelSpec::the()->getNetworkGevModel() ;
      return ;
    }
    err = new patErrMiscError("Unknown gev model") ;
    return ;
  }

}


void patBiogeme::checkBetaParameters(patError*& err) {
  patModelSpec::the()->checkAllBetaUsed(err) ;
  
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
}
void patBiogeme::readSampleFiles(patError*& err) {

  DEBUG_MESSAGE("Read data file") ;
  if (typeOfRun != patGeneratingCode) {
    theSample->readDataFile(theNormalGenerator,theRectangularGenerator,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
    result.numberOfObservations = theSample->getNumberOfObservations() ;
    result.numberOfIndividuals = theSample->getNumberOfIndividuals() ;
    result.cteLikelihood = theSample->computeLogLikeWithCte() ;
    DETAILED_MESSAGE("Nbr of attributes: " << patModelSpec::the()->getNbrAttributes()) ;
    DETAILED_MESSAGE("Nbr of alternatives: " << patModelSpec::the()->getNbrAlternatives()) ;
    DETAILED_MESSAGE("Sample size: " << theSample->getSampleSize()) ;
    DETAILED_MESSAGE("Nbr of groups: " << theSample->numberOfGroups()) ;
    DETAILED_MESSAGE("Nbr of betas: " << patModelSpec::the()->getNbrOrigBeta()) ;
//     DETAILED_MESSAGE("Nbr of betas (inc. Box-Cox): " 
// 		     << patModelSpec::the()->getNbrTotalBeta()) ;
    if (patModelSpec::the()->isMixedLogit()) {
      DETAILED_MESSAGE("Nbr of draws: " << patModelSpec::the()->getNumberOfDraws()) ;
    }
    
    if (theSample->getSampleSize() == 0) {
      err = new patErrMiscError("No data in the sample") ;
      WARNING(err->describe()) ;
      return ;
    }

  }
}


void patBiogeme::scaleUtilityFunctions(patError*& err) {

  util = patModelSpec::the()->getFullUtilityFunction(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  
  if (patParameters::the()->getgevAutomaticScalingOfLinearUtility() &&
      util->isLinear() ) {
    patModelSpec::the()->automaticScaling = patTRUE ;
    patVariables* lom = theSample->getLevelOfMagnitude(err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
    patModelSpec::the()->identifyScales(lom,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }
    theSample->scaleAttributes(patModelSpec::the()->getAttributesScale(),err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return ;
    }

    patModelSpec::the()->scaleBetaParameters() ;
  }
  
}
void patBiogeme::initProbaModel(patError*& err) {

  if (patModelSpec::the()->isMixedLogit() &&
      patModelSpec::the()->getNumberOfDraws() < patParameters::the()->getgevWarningLowDraws()) {
    GENERAL_MESSAGE("*************************************") ;
    GENERAL_MESSAGE("* The number of draws is low: " << 
		    patModelSpec::the()->getNumberOfDraws()) ; 
    GENERAL_MESSAGE("*************************************") ;
  }
  if (util == NULL) {
    util = patModelSpec::the()->getFullUtilityFunction(err) ;
  }
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (patModelSpec::the()->isBP()) {
    if (patModelSpec::the()->isPanelData()) {
      err = new patErrMiscError("No panel data with binary probit... Sorry.") ;
      WARNING(err->describe()) ;
      return ;
    }
    else {
      theModel = new patProbaProbitModel(util) ;
    }
  }
  else if (patModelSpec::the()->isOL()) {
    if (patModelSpec::the()->isPanelData()) {
      err = new patErrMiscError("No panel data with ordinal logit... Sorry.") ;
      WARNING(err->describe()) ;
      return ;
    }
    else {
      DEBUG_MESSAGE("About to create the ordinal logit model") ;
      theModel = new patProbaOrdinalLogit(util) ;
      DEBUG_MESSAGE("Done.") ;
    }
  }
  else if (patModelSpec::the()->isMNL()) {
    if (patModelSpec::the()->isPanelData()) {
      DEBUG_MESSAGE("MNL PANEL DATA") ;
      thePanelModel = new patProbaMnlPanelModel(util) ;
    }
    else {
      theModel = new patProbaMnlModel(util) ;
    }
  }
  else {
    if (patModelSpec::the()->isPanelData()) {
      thePanelModel = new patProbaGevPanelModel(gevModel,util) ;
    }
    else {
      theModel = new patProbaGevModel(gevModel,util) ;
    }
  }
  patString fileName("model.debug") ;
  ofstream debug(fileName) ;
  debug << *patModelSpec::the() << endl ;
  debug.close() ;
  patOutputFiles::the()->addDebugFile(fileName,"Debugging information");


}


patBoolean patBiogeme::initLikelihoodFunction(patError*& err) {

  getParameters() ;
  like = new patLikelihood(theModel,thePanelModel,theSample,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  switch (typeOfRun) {
  case patGeneratingCode : {
    if (!patModelSpec::the()->isMNL()) {
      err = new patErrMiscError("fastbiogeme can only be used with MNL specifications. Sorry.") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    // The source code must be created
    functionToMinimize = new patMinimizedFunction(like,theTrParameters) ;  
    functionToMinimize->setStopFileName(patParameters::the()->getgevStopFileName()) ;
    patString fileName = patFileNames::the()->getCcCode(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    ofstream cppFile(fileName.c_str()) ;
    
    patAbsTime now ;
    now.setTimeOfDay() ;
    cppFile << "// File name: " <<  fileName << endl ;
    cppFile << "// This file has automatically been generated." << endl ;
    cppFile << "// " << now.getTimeString(patTsfFULL) << endl ;
    cppFile << "// " << patVersion::the()->getCopyright() << endl ;
    cppFile << "// " << patVersion::the()->getVersionInfoDate() << endl ;
    cppFile << "// " << patVersion::the()->getVersionInfoAuthor() << endl ;
    cppFile << endl ;
    
    
    functionToMinimize->generateCppCode(cppFile, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    cppFile.close() ;
    patOutputFiles::the()->addDebugFile(fileName,"C++ specification of the model");
    GENERAL_MESSAGE("File " << fileName << " has been created") ;
    return patFALSE ;
  }
    break ;
  case patParallelRun: {
    // We use the function
    DEBUG_MESSAGE("WE USE USER DEFINED FUNCTION") ;
    functionToMinimize = theFastFunction ;
    if (functionToMinimize == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    functionToMinimize->setStopFileName(patParameters::the()->getgevStopFileName()) ;
  }
    break ;
  case patNormalRun: {
    DEBUG_MESSAGE("Original biogeme function"); 
    functionToMinimize = new patMinimizedFunction(like,theTrParameters) ;  
    functionToMinimize->setStopFileName(patParameters::the()->getgevStopFileName()) ;
  }
  }
  dim = functionToMinimize->getDimension() ;
  DETAILED_MESSAGE("Dimension of the optimisation problem = " << dim) ;
  return patTRUE ;
}


void patBiogeme::defineBounds(patError*& err) {

  bounds = new trBounds(dim) ;
  
  patVariables lower = patModelSpec::the()->getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  if (lower.size() != dim) {
    WARNING("Lower is " << lower.size()) ;
    WARNING("should be " << dim) ;
    return ;
  }
  patVariables upper = patModelSpec::the()->getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (upper.size() != dim) {
    WARNING("Lower is " << upper.size()) ;
    WARNING("should be " << dim) ;
    return ;
  }

  if (util != NULL) {
  DETAILED_MESSAGE("Utility function: " << util->getName()) ;
  }
  else {
    WARNING("Undefined utility function");
  }

  
  for (unsigned long i = 0 ;
       i < dim ;
       ++i) {
    
    bounds->setBounds(i,
		     lower[i],
		     upper[i],
		     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}

void patBiogeme::computeLikelihoodOfTrivialModel(patError*& err) {
  getParameters() ;
  if (functionToMinimize == NULL) {
    err = new patErrNullPointer("patMinimizedFunction") ;
    WARNING(err->describe()) ;
    return ;
  }
  //  trVector x0 = functionToMinimize->getCurrentVariables(err) ;
  trVector x0 = patModelSpec::the()->getEstimatedCoefficients(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patBoolean success ;

  DEBUG_MESSAGE("getTrivialModelLikelihood()") ;  
  result.nullLoglikelihood =  like->getTrivialModelLikelihood();
  DEBUG_MESSAGE("Null log likelihood: " << result.nullLoglikelihood) ;
  
  trVector analGrad(dim) ;
  trHessian theHessian(theTrParameters,dim) ;
  if (patParameters::the()->getBTRExactHessian() && functionToMinimize->isHessianAvailable()) {
    result.initLoglikelihood = 
      -functionToMinimize->computeFunctionAndDerivatives(&x0,&analGrad,&theHessian,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

  }
  else {
    result.initLoglikelihood = 
      -functionToMinimize->computeFunctionAndDerivatives(&x0,&analGrad,NULL,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }



  if (patParameters::the()->getgevCheckDerivatives()) {

    GENERAL_MESSAGE("Check derivatives...") ;
    trVector finDiffGrad(dim);
    patAbsTime beg ;
    patAbsTime endc ;
    beg.setTimeOfDay() ;
    functionToMinimize->computeFinDiffGradient(&x0,&finDiffGrad,&success,err) ;
    endc.setTimeOfDay() ;
    patTimeInterval ti0(beg,endc) ;
    GENERAL_MESSAGE("Run time for finite difference gradient: " << ti0.getLength()) ;
    GENERAL_MESSAGE("Analytical\t\tFin. diff.\tError\t   Name") ;
    for (unsigned int i = 0 ; i < analGrad.size() ; ++i) {
      patBoolean found ;
      patBetaLikeParameter theParam = patModelSpec::the()->getParameterFromIndex(i,&found) ;
      GENERAL_MESSAGE( setprecision(7) << setiosflags(ios::scientific|ios::showpos) << analGrad[i] << '\t' << finDiffGrad[i] << '\t' << setprecision(2) << 100*(analGrad[i]-finDiffGrad[i])/finDiffGrad[i]<<"% "  << theParam.name << '\t') ;
    }
    if (patParameters::the()->getBTRExactHessian()) {
      GENERAL_MESSAGE("Check second derivatives...") ;
      trHessian theFinDiffHessian(theTrParameters,dim) ;
      functionToMinimize->computeFinDiffHessian(&x0,&theFinDiffHessian,&success,err) ;
      GENERAL_MESSAGE("Finite difference:") ;
      stringstream str2 ;
      theFinDiffHessian.print(str2) ;
      GENERAL_MESSAGE(str2.str()) ;
      GENERAL_MESSAGE("Analytical:") ;
      stringstream str1 ;
      theHessian.print(str1) ;
      GENERAL_MESSAGE(str1.str()) ;
    }
    
  }

  patParameters::the()->setBTRTypf(result.initLoglikelihood) ;
  GENERAL_MESSAGE("Init loglike=" << result.initLoglikelihood) ;

}






void patBiogeme::addConstraints(patError*& err) {

  getParameters() ;
  //  trVector x0 = functionToMinimize->getCurrentVariables(err) ;

  trVector x0 = patModelSpec::the()->getEstimatedCoefficients(err) ;
  patBoolean success ;
  theProblem = new patMaxLikeProblem(functionToMinimize,
				     bounds,
				     theTrParameters) ;
    
  // Add user defined linear equality constraints
  patListProblemLinearConstraint eqCons = 
    patModelSpec::the()->getLinearEqualityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
  for (patListProblemLinearConstraint::iterator iter = eqCons.begin() ;
       iter != eqCons.end() ;
       ++iter) {     
    DEBUG_MESSAGE(patModelSpec::the()->printEqConstraint(*iter)) ;
    theProblem->addLinEq(iter->first,iter->second) ;
  }

  // Add user defined linear inequality constraints
  patListProblemLinearConstraint ineqCons = 
    patModelSpec::the()->getLinearInequalityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
  for (patListProblemLinearConstraint::iterator iter = ineqCons.begin() ;
       iter != ineqCons.end() ;
       ++iter) {     
    DEBUG_MESSAGE(patModelSpec::the()->printIneqConstraint(*iter)) ;
    theProblem->addLinIneq(iter->first,iter->second) ;
  }
    

  // Add user defined nonlinear equality constraints
  patListNonLinearConstraints* nlEqCons =
    patModelSpec::the()->getNonLinearEqualityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (patParameters::the()->getgevCheckDerivatives()) {
    GENERAL_MESSAGE("Check derivatives of the constraints...") ;
  }
  if (nlEqCons != NULL) {
    for (patListNonLinearConstraints::iterator i = nlEqCons->begin() ;
	 i != nlEqCons->end() ;
	 ++i) {
      if (patParameters::the()->getgevCheckDerivatives()) {
	  
	GENERAL_MESSAGE("Analytical derivatives:") ;
	trVector tmp(dim) ;
	i->computeFunctionAndDerivatives(&x0,&tmp,NULL,&success,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	GENERAL_MESSAGE(tmp) ;
	trVector finDiffGrad(x0.size()) ;
	i->computeFinDiffGradient(&x0,&finDiffGrad,&success,err) ;
	GENERAL_MESSAGE("Fin diff deriv") ;
	GENERAL_MESSAGE(finDiffGrad) ;
      }	
      theProblem->addNonLinEq(&(*i)) ;
    }
  }

  // Add user defined nonlinear inequality constraints
  patListNonLinearConstraints* nlIneqCons =
    patModelSpec::the()->getNonLinearInequalityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (nlIneqCons != NULL) {
    for (patListNonLinearConstraints::iterator i = nlIneqCons->begin() ;
	 i != nlIneqCons->end() ;
	 ++i) {
	
      DEBUG_MESSAGE("Check derivatives...") ;
      DEBUG_MESSAGE("Analytical derivatives:") ;
      trVector tmp(x0.size()) ;
      i->computeFunctionAndDerivatives(&x0,&tmp,NULL,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      cout << tmp << endl ;
      trVector finDiffGrad(x0.size()) ;
      i->computeFinDiffGradient(&x0,&finDiffGrad,&success,err) ;
      DEBUG_MESSAGE("Fin diff deriv") ;
      cout << finDiffGrad << endl  ;
	
      theProblem->addNonLinIneq(&(*i)) ;
	
    }
  }

  // Add constraint on nest coefficients
    
  patIterator<pair<unsigned long, unsigned long> >* iter =
    patModelSpec::the()->createConstraintNestIterator(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  unsigned long n = patModelSpec::the()->getNbrNonFixedParameters() ;
  for (iter->first() ;
       !iter->isDone() ;
       iter->next()) {
    patVariables constraint(n,0.0) ;
    pair<unsigned long, unsigned long> ci = iter->currentItem() ;
    if (ci.first >= n) {
      err = new patErrOutOfRange<unsigned long>(ci.first,0,n-1) ;
      WARNING(err->describe()) ;
      return ;
    }
    if (ci.second >= n) {
      err = new patErrOutOfRange<unsigned long>(ci.second,0,n-1) ;
      WARNING(err->describe()) ;
      return ;
    }
    constraint[ci.first] = 1.0 ;
    constraint[ci.second] = -1.0 ;
    theProblem->addLinEq(constraint,0.0) ;
  }      

  if (!theProblem->isFeasible(x0,err)) {
    WARNING("Starting point not feasible") ;
  }

}

void patBiogeme::initAlgorithm(patError*& err) {
  getParameters() ;
  //  trVector x0 = functionToMinimize->getCurrentVariables(err) ;
  trVector x0 = patModelSpec::the()->getEstimatedCoefficients(err) ;
  if (patParameters::the()->getgevAlgo() == "BIOMC") {
    if (theProblem->nNonTrivialConstraints() > 0) {
      err = new patErrMiscError("The option gevAlgo=\"BIO\" cannot be used in the presence\n\tof non trivial constraints on the parameters.\n\tUse gevAlgo=\"CFSQP\"") ;
      WARNING(err->describe()) ;
      return ;
    }
    if (patModelSpec::the()->isMixedLogit()) {
      theSimBasedProblem = 
	new patSimBasedMaxLikeOptimization(theProblem,
					   patModelSpec::the()->getNumberOfDraws()) ;
      algo = new trSimBasedSimpleBoundsAlgo(theSimBasedProblem,
					    x0,
					    theTrParameters,
					    err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patIterationBackup* aBackup = new patBiogemeIterationBackup() ;
      algo->setBackup(aBackup) ;

    }
    else {
      patIterationBackup* aBackup = new patBiogemeIterationBackup() ;
      algo = new trSimpleBoundsAlgo(theProblem,
				    x0,
				    theTrParameters,
				    aBackup,
				    err) ;
      
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      
    }
  }  

  if (patParameters::the()->getgevAlgo() == "BIO") {
      
    if (theProblem->nNonTrivialConstraints() > 0) {
      err = new patErrMiscError("The option gevAlgo=\"BIO\" cannot be used in the presence\n\tof non trivial constraints on the parameters.\n\tUse gevAlgo=\"CFSQP\"") ;
      return ;
    }
    patIterationBackup* aBackup = new patBiogemeIterationBackup() ;
    algo = new trSimpleBoundsAlgo(theProblem,
				  x0,
				  theTrParameters,
				  aBackup,
				  err) ;
      
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
    

  else if (patParameters::the()->getgevAlgo() == "CFSQP") {
      
    DETAILED_MESSAGE("Use CFSQP") ;
      
    patIterationBackup* aBackup = new patBiogemeIterationBackup() ;
    patCfsqp* cfsqpAlgo = new patCfsqp(aBackup,theProblem) ;
    cfsqpAlgo->setParameters(patParameters::the()->getcfsqpMode(),
			     patParameters::the()->getcfsqpIprint(),
			     patParameters::the()->getcfsqpMaxIter(),
			     patParameters::the()->getcfsqpEps(),
			     patParameters::the()->getcfsqpEpsEqn(),
			     patParameters::the()->getcfsqpUdelta(),
			     patParameters::the()->getgevStopFileName()) ;
    algo = cfsqpAlgo ;
    if (algo == NULL) {
      err = new patErrNullPointer("patCfsqp") ;
      return ;
    }
    algo->defineStartingPoint(x0) ;
  }
  else if (patParameters::the()->getgevAlgo() == "SOLVOPT") {
    DEBUG_MESSAGE("Use SolvOpt algorithm") ;
    algo = new patSolvOpt(theSolvoptParameters,theProblem) ;
    if (algo == NULL) {
      err = new patErrNullPointer("patSolvOpt") ;
      WARNING(err->describe()) ;
      return ;
    }
    algo->defineStartingPoint(x0) ;
  }
  else if (patParameters::the()->getgevAlgo() == "DONLP2") {
    err = new patErrMiscError("Algorithm DONLP2 is not supported anymore");
    WARNING(err->describe()) ;
    return ;     
  }
  else {
    stringstream str ;
    str << "Algorithm " << patParameters::the()->getgevAlgo() << " is not supported by this version of Biogeme" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;     
  }

  if (!algo->isAvailable()) {
    stringstream str ;
    str << "Algorithm " << algo->getName() << " is not available" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  if (algo == NULL) {
    err = new patErrMiscError("No algorithm specified") ;
    WARNING(err->describe()) ;
    return ;     
  }
}

void patBiogeme::runAlgorithm(patError*& err) {
  patAbsTime beg ;
  patAbsTime endc ;

  beg.setTimeOfDay() ;

  patString algoName(algo->getName()) ;
  
  DEBUG_MESSAGE("About to run algorithm: " << algoName) ;

  patString diag = algo->run(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  endc.setTimeOfDay() ;
  patTimeInterval ti0(beg,endc) ;
  DEBUG_MESSAGE("Run time interval: " << ti0 ) ;
  result.runTime = ti0.getLength() ;
  result.diagnostic = diag ;
  result.iterations = algo->nbrIter() ;
  GENERAL_MESSAGE("Run time: " << result.runTime) ;
}
void patBiogeme::analyzeResults(patPythonResults* pythonRes, patError*& err) {

  patBoolean success ;
  patVariables solution = algo->getSolution(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
  patReal valueSolution = algo->getValueSolution(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
    
  patModelSpec::the()->setEstimatedCoefficients(&solution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


    
  //      patVariables lowerBoundsLagrange = algo->getLowerBoundsLambda() ;
  //      DEBUG_MESSAGE("Lagrange LB " << lowerBoundsLagrange) ;
  //      patVariables upperBoundsLagrange = algo->getUpperBoundsLambda() ;
  //      DEBUG_MESSAGE("Lagrange UP " << upperBoundsLagrange) ;
  //    patBoolean success ;
  trVector finalGrad(solution.size()) ;

  functionToMinimize->computeFunctionAndDerivatives(&solution,&finalGrad,NULL,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  


  if (solution.size() < 11) {
    DETAILED_MESSAGE("Solution = " << solution) ;
    DETAILED_MESSAGE("Gradient = " << finalGrad) ;
  }

  patListNonLinearConstraints* nlEqCons =
    patModelSpec::the()->getNonLinearEqualityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  if (nlEqCons != NULL) {
    DETAILED_MESSAGE("Value of the nonlinear equality constraints") ;
    for (patListNonLinearConstraints::iterator i = nlEqCons->begin() ;
	 i != nlEqCons->end() ;
	 ++i) {
      DETAILED_MESSAGE(i->computeFunction(&solution,&success,err) << "=" << *i) ;
    }
  }
  patListNonLinearConstraints* nlIneqCons =
    patModelSpec::the()->getNonLinearInequalityConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }
  if (nlIneqCons != NULL) {
    DETAILED_MESSAGE("Value of the nonlinear inequality constraints") ;
    for (patListNonLinearConstraints::iterator i = nlIneqCons->begin() ;
	 i != nlIneqCons->end() ;
	 ++i) {
      DETAILED_MESSAGE(*i << "=" << i->computeFunction(&solution,&success,err)) ;
    }
  }


  GENERAL_MESSAGE("Final log likelihood=" << -valueSolution) ;

  GENERAL_MESSAGE("Be patient... BIOGEME is preparing the output files") ;

  result.loglikelihood = -valueSolution  ;
  result.gradientNorm = 
    sqrt(inner_product(finalGrad.begin(),
		       finalGrad.end(),
		       finalGrad.begin(),0.0)) ;

  patBoolean isSolutionOK(patTRUE) ;
  for (unsigned long k = 0 ;  k < solution.size() ; ++k) {
    if (!(isfinite(solution[k]) && isfinite(finalGrad[k]))) {
      isSolutionOK = patFALSE ;
      DEBUG_MESSAGE("Solution: " << solution) ;
      DEBUG_MESSAGE("Gradient: " << finalGrad) ;
      err = new patErrMiscError("Serious numerical problem: the model cannot be evaluated at the final value of the coefficients") ;
      WARNING(err->describe()) ;
      return ;
    }
  }

  patMyMatrix* robustPtr(NULL) ;
  if (isSolutionOK) {
    unsigned long varCovarSize = theProblem->getSizeOfVarCovar() ;
    varCovar = new patMyMatrix(varCovarSize,varCovarSize) ; 
    robustVarCovar = new patMyMatrix(dim,dim) ; 

    if (!patFileExists()(patParameters::the()->getgevStopFileName())) {
      
      DEBUG_MESSAGE("Compute variance-covariance matrix") ;
      
      //     success = theProblem->computeSimpleVarCovar(&solution,
      // 					       &varCovar,
      // 					       &robustVarCovar,
      // 					       &(result.eigenVector),
      // 					       &(result.Az),
      // 					       err) ;
      
      
      unsigned long nActive = theProblem->getNumberOfActiveConstraints(solution,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      
      DEBUG_MESSAGE("Number of active constraints at the solution: " << nActive) ;
      robustPtr = (nActive == 0)  ? robustVarCovar : NULL ;
      
      patAbsTime begVar ;
      patAbsTime endcVar ;
      
      begVar.setTimeOfDay() ;
      
      success = theProblem->computeVarCovar(&solution,
					    varCovar,
					    robustPtr,
					    &result.eigenVectors,
					    &result.smallestSingularValue,
					    err) ;

      endcVar.setTimeOfDay() ;
      
      patTimeInterval ti0Var(begVar,endcVar) ;
      DEBUG_MESSAGE("Run time interval: " << ti0Var ) ;
      GENERAL_MESSAGE("Run time for var/covar computation: " << ti0Var.getLength()) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    else {
      WARNING("BIOGEME has been interrupted by the user.\n\tThe variance-covariance matrix is not computed") ;
      success = patFALSE ;
    }
  }

  result.varCovarMatrix = varCovar ;
  result.robustVarCovarMatrix = robustVarCovar ;
  if (success) {
    result.isVarCovarAvailable = patTRUE ;
    result.isRobustVarCovarAvailable = (robustPtr != NULL) ;
  }
  
  if (patModelSpec::the()->automaticScaling) {
    patModelSpec::the()->unscaleBetaParameters() ;
    if (result.isVarCovarAvailable) {
      patModelSpec::the()->unscaleMatrix(result.varCovarMatrix,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    if (result.isRobustVarCovarAvailable) {
      patModelSpec::the()->unscaleMatrix(result.robustVarCovarMatrix,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }


  DEBUG_MESSAGE("Transfert results") ;
  patModelSpec::the()->setEstimationResults(result) ;
  DEBUG_MESSAGE("Done.") ;
  

  patString resFile = patFileNames::the()->getResFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;

  }
  patModelSpec::the()->writeSpecFile(resFile,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patString enuFile = patFileNames::the()->getEnuFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  
  patString repFile = patFileNames::the()->getRepFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patModelSpec::the()->writeReport(repFile,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patString htmlFile = patFileNames::the()->getHtmlFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patModelSpec::the()->writeHtml(htmlFile,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (patParameters::the()->getgevGeneratePythonFile() != 0) {
    patString pythonSpecFile = patFileNames::the()->getPythonSpecFile(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  
    patModelSpec::the()->writePythonSpecFile(pythonSpecFile,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  patString latexFile = patFileNames::the()->getLatexFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patModelSpec::the()->writeLatex(latexFile,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  if (pythonRes != NULL) {
    patModelSpec::the()->writePythonResults(pythonRes,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  if (patParameters::the()->getgevGenerateFilesForDenis() != 0) {
    patString fileForDenis = patParameters::the()->getgevFileForDenis() ;
    patModelSpec::the()->generateFileForDenis(fileForDenis) ;
  }

}

void patBiogeme::finalMessages(patError*& err) {
  GENERAL_MESSAGE("BIOGEME Input files") ;
  GENERAL_MESSAGE("===================") ;
  GENERAL_MESSAGE("Parameters:\t\t\t" << patFileNames::the()->getParFile()) ;
  GENERAL_MESSAGE("Model specification:\t\t" << patFileNames::the()->getModFile()) ;
  unsigned short nbrSampleFiles = patFileNames::the()->getNbrSampleFiles() ;
  for (unsigned short fileId = 0 ; fileId < nbrSampleFiles ; ++fileId) {
    patString fileName = patFileNames::the()->getSamFile(fileId,err) ;
    GENERAL_MESSAGE("Sample " << 1+fileId << " :\t\t\t\t" << fileName) ;
  }

  if (theModel != NULL) {
    GENERAL_MESSAGE("Model informations: " << theModel->getModelName(err)) ;
    GENERAL_MESSAGE("==================") ;
    GENERAL_MESSAGE(theModel->getInfo()) ;
  }


  //  DEBUG_MESSAGE("Profiling information") ;
  //patTimer::the()->generateReport() ;

}

void patBiogeme::sampleEnumeration(patPythonReal** arrayResult,
				   unsigned long resRow,
				   unsigned long resCol,
				   patError*& err) {

  patString enuFile = patFileNames::the()->getEnuFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patUnixUniform pseudo(patParameters::the()->getgevSeed()) ;
  if (theModel == NULL) {
    if (thePanelModel != NULL) {
      err = new patErrMiscError("The [PanelData] section must be removed for simulation") ;
      
    } 
    else {
      err = new patErrMiscError("The model specification has not been loaded properly.") ;
    }
    WARNING(err->describe()) ;
    return ;
  }

  unsigned short nbrOfZf(0) ;
  unsigned short nbrOfProbaZf(0) ;
  nbrOfZf = patModelSpec::the()->numberOfZhengFosgerau(&nbrOfProbaZf) ;
  enuIndices = 
    new patSampleEnuGetIndices(patModelSpec::the()->includeUtilitiesInSimulation(),
			       patModelSpec::the()->getNbrAlternatives(),
			       patMax(0,nbrOfZf-nbrOfProbaZf)) ;
  if (arrayResult == NULL &&  nbrOfZf > 0) {
  
    resCol = enuIndices->getNbrOfColumns() ;
    resRow = theSample->getSampleSize() ;
    arrayResult = new patPythonReal*[resRow] ;
    for (unsigned short j = 0 ; j < resRow ; ++j) {
      arrayResult[j] = new patPythonReal[resCol] ;
    }
  }


  
  patSampleEnumeration sampleEnum(enuFile,
				  arrayResult,
				  resRow,
				  resCol,
				  theSample,
				  theModel,
				  enuIndices,
				  &pseudo) ;
  sampleEnum.enumerate(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  if (nbrOfZf > 0) {

    patZhengFosgerau 
      theZhengFosgerauTest(arrayResult,
			   resRow,
			   enuIndices,
			   patModelSpec::the()->getZhengFosgerauVariables(),
			   err) ;

    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    theZhengFosgerauTest.compute(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    patString fileName = patFileNames::the()->getZhengFosgerauLatex(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    theZhengFosgerauTest.writeLatexReport(fileName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    fileName = patFileNames::the()->getZhengFosgerau(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    theZhengFosgerauTest.writeExcelReport(fileName,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
    


  if (patParameters::the()->getgevGenerateGnuplotFile() != 0) {
    patString gnuplotFile = patFileNames::the()->getGnuplotFile(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    patModelSpec::the()->writeGnuplotFile(gnuplotFile,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  
    
  }


  GENERAL_MESSAGE("BIOSIM Input file") ;
  GENERAL_MESSAGE("=================") ;
  GENERAL_MESSAGE("Model specification:\t" << patFileNames::the()->getModFile()) ;
}

void patBiogeme::loadModelAndSample(patError*& err) {
  initLogFiles(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  theSample = new patSample ;

  if (dataArray == NULL) {
    checkSampleFiles(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    readDataHeaders(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
  else {
    theSample->externalDataStructure(dataArray,nRows,nColumns) ;
    patModelSpec::the()->setDataHeader(headers) ;
  }

  if (theFastFunction != NULL) {
    theFastFunction->setSample(theSample);
  }

  patBoolean mustContinue = readModelSpecification(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (!mustContinue) {
    return ;
  }
  
  if (typeOfRun != patNormalRun && patModelSpec::the()->utilDerivativesAvailableFromUser() ) {

    err = new patErrMiscError("When biogeme is not used in normal mode, please do not include user-defined derivatives. Please remove the section [Derivatives] and start again.") ;
    WARNING(err->describe()) ;
    return ;
  }

  initRandomNumberGenerators(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  readSummaryParameters(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  defineGevModel(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  checkBetaParameters(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  readSampleFiles(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  scaleUtilityFunctions(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  initProbaModel(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}



void patBiogeme::estimate(patPythonResults* pythonRes, patError*& err) {

  patBoolean canContinue = initLikelihoodFunction(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (!canContinue) {
    return ;
  }

  defineBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  computeLikelihoodOfTrivialModel(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  addConstraints(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  initAlgorithm(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  runAlgorithm(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  analyzeResults(pythonRes,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  finalMessages(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}


void patBiogeme::externalData(  patPythonReal** d, 
				unsigned long nr, 
				unsigned long nc, 
				vector<patString> head ) {
  dataArray = d ;
  nRows = nr ;
  nColumns = nc ;
  headers = head ;
}

void patBiogeme::getParameters() {
  theTrParameters.eta1 =  patParameters::the()->getBTREta1() ;
  theTrParameters.eta2 =  patParameters::the()->getBTREta2() ; ;
  theTrParameters.gamma2 = patParameters::the()->getBTRGamma2() ; ;
  theTrParameters.beta = patParameters::the()->getBTRIncreaseTRRadius() ; ;
  theTrParameters.maxIter =  patParameters::the()->getBTRMaxIter() ; ;
  theTrParameters.initQuasiNewtonWithTrueHessian = patParameters::the()->getBTRInitQuasiNewtonWithTrueHessian() ;
  theTrParameters.initQuasiNewtonWithBHHH = patParameters::the()->getBTRInitQuasiNewtonWithBHHH() ;
  theTrParameters.significantDigits = patParameters::the()->getBTRSignificantDigits(); ;
  theTrParameters.usePreconditioner = patParameters::the()->getBTRUsePreconditioner(); ;
  theTrParameters.maxTrustRegionRadius = patParameters::the()->getBTRMaxTRRadius() ;
  theTrParameters.typicalF = patParameters::the()->getBTRTypf() ;
  theTrParameters.tolerance = patParameters::the()->getBTRTolerance() ;
  theTrParameters.toleranceSchnabelEskow =  patParameters::the()->getTolSchnabelEskow() ;
  theTrParameters.exactHessian = patParameters::the()->getBTRExactHessian() ;
  theTrParameters.cheapHessian = patParameters::the()->getBTRCheapHessian() ;
  theTrParameters.initRadius = patParameters::the()->getBTRInitRadius() ;
  theTrParameters.minRadius = patParameters::the()->getBTRMinTRRadius() ;
  theTrParameters.armijoBeta1 = patParameters::the()->getBTRArmijoBeta1() ;
  theTrParameters.armijoBeta2 = patParameters::the()->getBTRArmijoBeta2() ;
  theTrParameters.stopFileName = patParameters::the()->getgevStopFileName() ;
  theTrParameters.startDraws =  patParameters::the()->getBTRStartDraws() ; ;
  theTrParameters.increaseDraws =  patParameters::the()->getBTRIncreaseDraws() ; ;
  theTrParameters.maxGcpIter = patParameters::the()->getBTRMaxGcpIter() ;
  theTrParameters.fractionGradientRequired =  patParameters::the()->getTSFractionGradientRequired() ;
  theTrParameters.expTheta = patParameters::the()->getTSExpTheta() ;
  theTrParameters.infeasibleCgIter = patParameters::the()->getBTRUnfeasibleCGIterations() ;
  theTrParameters.quasiNewtonUpdate =  patParameters::the()->getBTRQuasiNewtonUpdate() ;
  theTrParameters.kappaUbs =  patParameters::the()->getBTRKappaUbs() ;
  theTrParameters.kappaLbs =  patParameters::the()->getBTRKappaLbs() ;
  theTrParameters.kappaFrd =   patParameters::the()->getBTRKappaFrd() ;
  theTrParameters.kappaEpp =  patParameters::the()->getBTRKappaEpp()  ;
  
  theSolvoptParameters.errorArgument = patParameters::the()->getsolvoptErrorArgument() ;
  theSolvoptParameters.errorFunction = patParameters::the()->getsolvoptErrorFunction() ;
  theSolvoptParameters.maxIter =  patParameters::the()->getsolvoptMaxIter() ;
  theSolvoptParameters.display =  patParameters::the()->getsolvoptDisplay() ;
  
  // theDonlp2Parameters.epsx = patParameters::the()->getdonlp2Epsx() ;
  // theDonlp2Parameters.delmin = patParameters::the()->getdonlp2Delmin() ;
  // theDonlp2Parameters.smallw = patParameters::the()->getdonlp2Smallw() ;
  // theDonlp2Parameters.epsdif = patParameters::the()->getdonlp2Epsdif() ;
  // theDonlp2Parameters.nreset = patParameters::the()->getdonlp2NReset() ;
  // theDonlp2Parameters.stopFileName = patParameters::the()->getgevStopFileName() 
    ;


}
