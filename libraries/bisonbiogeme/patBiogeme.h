//-*-c++-*------------------------------------------------------------
//
// File name : patBiogeme.h
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Sun Apr 10 14:18:42 2005
//
//--------------------------------------------------------------------

class patError ;
class patRandomNumberGenerator ;
class patGEV ;
class patUtility ;
class patProbaModel ;
class patProbaPanelModel ;
class patLikelihood ;
class giaLikelihood ;
class patMinimizedFunction ;
class patGianlucaFunction ;
class trBounds ;
class trNonLinearAlgo ;
class patMaxLikeProblem ;
class patSimBasedMaxLikeOptimization ;
class trFunction ;
class patPythonResults ;
class patFastBiogeme ;
class patSampleEnuGetIndices ;

#ifndef patBiogeme_h
#define patBiogeme_h

#include "patSample.h"
#include "patEstimationResult.h"
#include "patUnixUniform.h"
#include "patString.h"
#include "patString.h"
#include "patBiogemeRun.h"
#include "trParameters.h"
#include "solvoptParameters.h"

/**
   @doc This is the general class enabling to access the main functionalitis of Biogeme
   @author \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}, EPFL (Sun Apr 10 14:18:42 2005)
*/

class patBiogeme {
 public:
  patBiogeme()  ;
  ~patBiogeme()  ;

  void loadModelAndSample(patError*& err) ;
  void loadModelAndSampleForGianluca(patError*& err) ;
  void estimate(patPythonResults* pythonRes, patError*& err) ;
  void sampleEnumeration(patPythonReal** arrayResult,
			 unsigned long resRow,
			 unsigned long resCol,
			 patError*& err) ;

public:
  void readParameterFile(patError*& err) ;
  void checkSampleFiles(patError*& err) ;
  void initLogFiles(patError*& err) ;
  void readDataHeaders(patError*& err) ;
  patBoolean readModelSpecification(patError*& err) ;
  void initRandomNumberGenerators(patError*& err) ;
  void readSummaryParameters(patError*& err) ;
  void defineGevModel(patError*& err) ;
  void checkBetaParameters(patError*& err) ;
  void readSampleFiles(patError*& err) ;
  void scaleUtilityFunctions(patError*& err) ;
  void initProbaModel(patError*& err) ;

  /**
     @return patTRUE if we can continue, and patFALSE if a C++ file has been created, and no further comnputation is requested.
   */
  patBoolean initLikelihoodFunction(patError*& err) ;
  void initGianlucaLikelihoodFunction(patError*& err) ;
  void defineBounds(patError*& err) ;
  void computeLikelihoodOfTrivialModel(patError*& err) ;
  void computeGianlucaLikelihoodOfTrivialModel(patError*& err) ;
  void addConstraints(patError*& err) ;
  void initAlgorithm(patError*& err) ;
  void runAlgorithm(patError*& err) ;
  void analyzeResults(patPythonResults* pythonRes, patError*& err) ;
  void finalMessages(patError*& err) ;
  void externalData(  patPythonReal** d, 
		      unsigned long nr, 
		      unsigned long nc, 
		      vector<patString> headers ) ;

public:
  patBiogemeRun typeOfRun ;
  patFastBiogeme* theFastFunction ;

private:
  void getParameters() ;
 private:

  patString debugSpecFile ;
  patEstimationResult result ;
  patRandomNumberGenerator* theNormalGenerator ;
  patRandomNumberGenerator* theRectangularGenerator ;
  patUniform* rng ;
  patGEV* gevModel ;
  patSample* theSample ;
  patUtility* util ;
  patProbaModel* theModel ;
  patProbaPanelModel* thePanelModel ;
  patLikelihood* like ;
  giaLikelihood* gianlucaLike ;
  trFunction* functionToMinimize ;
  unsigned long dim ;
  trBounds* bounds ;
  trNonLinearAlgo* algo;
  patMaxLikeProblem* theProblem ;
  patSimBasedMaxLikeOptimization*  theSimBasedProblem ;
  vector<trFunction*> constraints ;
  patMyMatrix* varCovar ;
  patMyMatrix* robustVarCovar ;


  patPythonReal** dataArray ;
  unsigned long nRows ;
  unsigned long nColumns ;
  vector<patString> headers ;
 
  patSampleEnuGetIndices* enuIndices ;

  trParameters theTrParameters ;
  solvoptParameters theSolvoptParameters ;
};

#endif
