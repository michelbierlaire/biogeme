//-*-c++-*------------------------------------------------------------
//
// File name : patBiogemeScripting.cc
// Author :    \URL[Michel Bierlaire]{http://roso.epfl.ch/mbi}
// Date :      Tue Dec 13 15:57:07 2005
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdlib.h>
#include "patBiogemeScripting.h"
#include "patFileNames.h"
#include "patVersion.h"
#include "patCfsqp.h"
#include "patDisplay.h"
#include "patParameters.h"
#include "patTimeInterval.h"
#include "patSingletonFactory.h"
#include "patBisonSingletonFactory.h"
#include "patOutputFiles.h"

patBiogemeScripting::patBiogemeScripting()  {
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;
  cout << patVersion::the()->getVersionInfoDate() << endl ;
  cout << patVersion::the()->getVersionInfoAuthor() << endl ;
  cout << patVersion::the()->getVersionInfoCompiled() << endl ;
  cout << "See http://biogeme.epfl.ch" << endl  ;
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl  ;
#ifdef IPOPT
  cout << "Optimization algorithm IPOPT available" << endl  ;
#else
  cout << "Optimization algorithm IPOPT not available" << endl  ;
#endif
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl  ;
  cout << "\t\"In every non-trivial program there is at least one bug.\"" << endl  ;
  cout << "" << endl  ;
  patOutputFiles::the()->clearList() ;

}

void patBiogemeScripting::estimate(int argc, char *argv[]) {
  //%%%%%%%%%%%%%%%%%%
  // 1 Test the arguments
  //%%%%%%%%%%%%%%%%%%

  if ((argc == 2) && patString(argv[1]) == patString("-h")) {
    cout << "Usage: " << argv[0] << " model_name sampleFile1 sampleFile2 sampleFile3 ... " << endl ;
    exit(0 );
  }

  //%%%%%%%%%%%%%%%%%%
  // 2 Define the model name
  //%%%%%%%%%%%%%%%%%%

  
  if (argc > 1) {
    patString modelName(argv[1]) ;
    patFileNames::the()->setModelName(modelName) ;
  }

  //%%%%%%%%%%%%%%%%%%
  // 3 Define the name of the sample files
  //%%%%%%%%%%%%%%%%%%

  if (argc > 2) {
    for (unsigned short i=2 ; i < argc ; ++i) {
      patString sampleFileName(argv[i]) ;
      patFileNames::the()->addSamFile(sampleFileName) ;
    }
  }  
  
  cout << endl ;

  //%%%%%%%%%%%%%%%%%%
  // 5 Invoke biogeme
  //%%%%%%%%%%%%%%%%%%

  invokeBiogeme() ;

  // Clear memory
  // This generates an error. I don't know why. 
  // DEBUG_MESSAGE("FINISHED. CLEARING MEMORY") ;
  // patSingletonFactory::the()->reset() ;
  // patBisonSingletonFactory::the()->reset() ;
  // DEBUG_MESSAGE("CLEARING MEMORY: DONE") ;
  patOutputFiles::the()->display() ;
  
}

void patBiogemeScripting::simulation(int argc, char *argv[]) {


  //%%%%%%%%%%%%%%%%%%
  // S1 Test the arguments
  //%%%%%%%%%%%%%%%%%%


  patError* err = NULL ;
  if ((argc == 2) && patString(argv[1]) == patString("-h")) {
    cout << "Usage: " << argv[0] << " model_name sampleFile1 sampleFile2 sampleFile3 ... " << endl ;
    exit(0 );
  }

  if (argc > 1) {
    patString modelName(argv[1]) ;
    patFileNames::the()->setModelName(modelName) ;
  }
  if (argc > 2) {
    for (unsigned short i=2 ; i < argc ; ++i) {
      patString sampleFileName(argv[i]) ;
      patFileNames::the()->addSamFile(sampleFileName) ;
    }
  }  
  
  cout << endl ;

  //%%%%%%%%%%%%%%%%%%
  // S2 Messages
  //%%%%%%%%%%%%%%%%%%


  //%%%%%%%%%%%%%%%%%%
  // S3 Read parameter file
  //%%%%%%%%%%%%%%%%%%

  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }
  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }

  patOutputFiles::the()->display() ;

}


void patBiogemeScripting::mod2py(int argc, char *argv[]) {

  patError* err = NULL ;

  //%%%%%%%%%%%%%%%%%%
  // 1 Test the arguments
  //%%%%%%%%%%%%%%%%%%


  if ((argc == 2) && patString(argv[1]) == patString("-h")) {
    cout << "Usage: " << argv[0] << " model_name" << endl ;
    exit(0);
  }

  //%%%%%%%%%%%%%%%%%%
  // 2 Define the model name
  //%%%%%%%%%%%%%%%%%%

  
  if (argc > 1) {
    patString modelName = patString(argv[1]) ;
    patFileNames::the()->setModelName(modelName) ;
  }
  else {
    WARNING("Usage: " << argv[0] << " model_name") ;
    exit(-1) ;
  }

  //%%%%%%%%%%%%%%%%%%
  // 3 Define the name of the sample files
  //%%%%%%%%%%%%%%%%%%

  if (argc > 2) {
    WARNING("mod2py requires only one arguments. All others arguments are ignores") ;
  }  
  
  cout << endl ;

  patString tmpFileName("bio__default.dat") ;
  cout << "Data: " << tmpFileName << endl ;
  ofstream dataFile(tmpFileName.c_str()) ;
  dataFile << "FakeHeader" << endl ;
  dataFile.close() ;
  patFileNames::the()->addSamFile(tmpFileName) ;
  patBiogeme biogeme ;
  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  // Parameters

  patParameters::the()->setgevGeneratePythonFile(1);
  patParameters::the()->setgevPythonFileWithEstimatedParam (0);

  cout << "Load model and sample" << endl ;
  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
    WARNING(err->describe());
    return ;
  }

  patOutputFiles::the()->display() ;
  
}




void patBiogemeScripting::invokeBiogeme() {

  patError* err = NULL ;
  patAbsTime t1 ;
  patAbsTime t2 ;
  patAbsTime t3 ;
  t1.setTimeOfDay() ;

  biogeme.readParameterFile(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }
  biogeme.loadModelAndSample(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }
  t2.setTimeOfDay() ;
  GENERAL_MESSAGE("Run time for data processing: " << patTimeInterval(t1,t2).getLength()) ;

  biogeme.estimate(NULL,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    exit(-1) ;
  }

  t3.setTimeOfDay() ;
  GENERAL_MESSAGE("Run time for estimation:      " << patTimeInterval(t2,t3).getLength()) ;
  GENERAL_MESSAGE("Total run time:               " << patTimeInterval(t1,t3).getLength()) ;
  if (biogeme.typeOfRun == patParallelRun) {
    GENERAL_MESSAGE("---- Run complete with " << patParameters::the()->getgevNumberOfThreads() << " processors ----") ;
  }
}


