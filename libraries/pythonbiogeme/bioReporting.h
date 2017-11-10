//-*-c++-*------------------------------------------------------------
//
// File name : bioReporting.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Aug 17 18:47:55 2009
//
//--------------------------------------------------------------------

#ifndef bioReporting_h
#define bioReporting_h

#include "patError.h"
#include "patMyMatrix.h"

class bioSimulatedValues ;
class bioRawResults ;
class bioBayesianResults ;
class bioSample ;
class patStatistics ;
class patKalman ;
class bioReporting {

 public:
  bioReporting(patError*& err) ;
  void writeHtml(patString fileName, patError*& err) ;
  void writeLaTeX(patString fileName, patError*& err) ;
  void writeALogit(patString fileName, patError*& err) ;
  patBoolean isRaoCramerAvailable() const ;
  patBoolean isSandwichAvailable() const ;
  void computeFromSample(bioSample* s, patError*& err)  ;
  void computeEstimationResults(bioRawResults* rr,patError*& err) ;
  void computeBayesianResults(bioBayesianResults* rr,patError*& err) ;
  void setDiagnostic(patString diag) ;
  void setIterations(patULong i) ;
  void setRunTime(patString t) ;
  void setDataProcessingTime(patString t) ;
  void setInitLL(patReal l) ;
  void addStatistic(patString name, patReal value) ;
  void addFormula(patString name, patString formula) ;
  void setSimulationResults(map<patString,bioSimulatedValues >* s, 
			    patBoolean sensitivity) ;
  void printEstimatedParameters(patString filename, patError*& err) ;
  void addMonteCarloReport(patStatistics* mainDraws,
				       patStatistics* cvDraws,
				       patKalman* theFilter,
				       patReal analytical, patError*& err) ;
  void involvesMonteCarlo() ;
  void setAlgorithm(patString a) ;
private:
  void compute(patError*& err) ;
  patBoolean reportDraws ;
  bioRawResults* theRawResults ;
  bioBayesianResults* theBayesianResults ;
  patVariables stdErr ;
  patVariables robustStdErr ;
  patULong nbrDraws ;
  patULong nbrParameters ;
  patULong sampleSize ;
  patULong removedObservations ;
  patReal initLoglikelihood ;
  patString algorithm ; 
  //  patReal nullLoglikelihood ;
  //  patReal cteLogLikelihood ;
  patString diagnostic ;
  patULong iterations ;
  patString runTime ;
  patString dataProcessingTime ;
  patString sampleFile ;
  map<patString, patReal> statistics ;
  map<patString, patString> formulas ;
  map<patString,bioSimulatedValues >* simulatedValues ;
  vector<patReal> mcNoCorrection ;
  vector<patReal> mcCorrected ;
  vector<patReal> mcStdDev ;
  vector<patReal> mcCorrectedStdDev ;
  vector<patReal> mcSimulated ;
  vector<patReal> mcAnalytical ;
  patULong mcReporting ;
  patBoolean parameterRead ;
  patBoolean sensitivityAnalysis ;
};

#endif
