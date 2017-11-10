//-*-c++-*------------------------------------------------------------
//
// File name : patPythonResults.h
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Sat Jul 22 15:44:04 2006
//
//--------------------------------------------------------------------

#ifndef patPythonResults_h
#define patPythonResults_h

#include <map>
#include <vector>
#include "patString.h"

class patPythonResults {

 public:

  // Time when the results are created
  const char* getTimeStamp() ;
  patString timeStamp ;

  // Version of biogeme
  const char* getVersion() ;
  patString version ;

  // Model description as provided by the user in the .mod file
  const char* getDescription() ;
  patString description ;

  // Type of model 
  const char* getModel() ;
  patString model ;

  // Type of draws (Random / Halton / Hess-Train)
  const char* getDrawsType() ;
  patString drawsType ;

  int numberOfDraws ;

  int numberOfParameters ;

  int numberOfObservations ;

  int numberOfIndividuals;
  
  // Loglikelihood for the trivial model where all alternatives are
  // equally likely
  patReal nullLoglikelihood ;
  
  // Loglikelihood at the starting point of the estimation
  patReal initLoglikelihood ;

  patReal finalLoglikelihood ;  
  
  patReal likelihoodRatioTest ;

  patReal rhoSquare ;

  patReal rhoBarSquare ;

  patReal finalGradientNorm ;

  const char* getVarianceCovariance() ;
  patString varianceCovariance  ;

  int totalNumberOfParameters ;

  vector<patString> paramNames ;
  map<patString, unsigned int> paramIndex;
  vector<patReal> estimates ;
  vector<patReal> stdErr ;
  vector<patReal> tTest ;
  vector<patReal> pValue ;
  vector<patReal> stdErrRobust ;
  vector<patReal> tTestRobust ;
  vector<patReal> pValueRobust ;
  vector<int>   fixed ;
  vector<int>   distributed ;

  const char* getParamName(unsigned int index) ;
  patReal getEstimate(unsigned int index) ;
  patReal getEstimate(const char* name) ;
  patReal getStdErr(unsigned int index) ;
  patReal getStdErr(const char* name) ;
  patReal getTTest(unsigned int index) ;
  patReal getTTest(const char* name) ;
  patReal getPValue(unsigned int index) ;
  patReal getPValue(const char* name) ;
  patReal getStdErrRobust(unsigned int index) ;
  patReal getStdErrRobust(const char* name) ;
  patReal getTTestRobust(unsigned int index) ;
  patReal getTTestRobust(const char* name) ;

  patReal getPValueRobust(unsigned int index) ;
  patReal getPValueRobust(const char* name) ;
  // 1 if the parameter is fixed, 0  otherwise
  int   getFixed(unsigned int index) ;
  int   getFixed(const char* name) ;
  // 1 if the parameter has a discrete distribution, 0  otherwise
  int   getDistributed(unsigned int index) ;
  int   getDistributed(const char* name) ;

} ;

#endif
