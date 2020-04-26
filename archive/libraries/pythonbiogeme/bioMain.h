//-*-c++-*------------------------------------------------------------
//
// File name : bioMain.h
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 17 16:37:06 2009
//
//--------------------------------------------------------------------

#ifndef bioMain_h
#define bioMain_h

#include "bioPythonWrapper.h"
#include "bioConstraintWrapper.h"
#include "bioReporting.h"
#include "bioBayesianResults.h"

class bioSample ;
class bioModel ;
class bioStatistics ;
class bioConstraints ;
class bioExpressionRepository ;
#include "trParameters.h"
#include "solvoptParameters.h"


class bioMain {
 public:
  bioMain() ;
  ~bioMain() ;
  void setFunction(bioPythonWrapper* f) ;
  void setStatistics(bioStatistics* s) ;
  void setConstraints(bioConstraints* s) ;
  void run(int argc, char *argv[], patError*& err) ;
  void run(patString modelFileName, patString dataFileName, patError*& err) ;
  void estimate(patError*& err) ;
  void simulate(patError*& err) ;
  void bayesian(patError*& err) ;
  void statistics(patError*& err) ;

  void setParameters(patError*& err) ;
  patString getHtmlFile() const ;
  patString getLatexFile() const ;
  patString getPythonEstimated() const ;
  
 private:
  bioPythonWrapper* theFunction;
  bioSample* theSample ;
  bioModel* theModel ;
  bioStatistics* theStatistics ;
  bioConstraints* theConstraints ;
  bioRawResults* theRawResults ;
  bioReporting* theReport ;

  trParameters theTrParameters ;
  solvoptParameters theSolvoptParameters ;

  bioBayesianResults bayesianResults ;
  bioExpressionRepository* theRepository ;
  patString htmlFile ;
  patString latexFile ;
  patString pythonEstimated ;
};

#endif
