//-*-c++-*------------------------------------------------------------
//
// File name : bioSensitivityAnalysis.cc
// Author :    Michel Bierlaire
// Date :      Sun May  6 15:02:31 2012
//
//--------------------------------------------------------------------

#include "patDisplay.h"
#include "bioSensitivityAnalysis.h"
#include "patQuantiles.h"
#include "patErrMiscError.h"
#include "bioParameters.h"
#include "patOutputFiles.h"

bioSensitivityAnalysis::bioSensitivityAnalysis() {

}

void bioSensitivityAnalysis::addData(patString name,patReal value) {
  theSimulatedValues[name].push_back(value) ;
}

vector<patReal> bioSensitivityAnalysis::getQuantiles(patString name, 
						     vector<patReal> alphas, 
						     patError*& err) {
  vector<patReal> result ;
  map<patString,patVariables>::iterator found = theSimulatedValues.find(name) ;
  if (found == theSimulatedValues.end()) {
    stringstream str ;
    str << "Simulated quantitiy " << name << " not found" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return result ;
  }
  patQuantiles theQuantiles(&found->second) ;
  for (vector<patReal>::iterator i = alphas.begin() ;
       i != alphas.end() ;
       ++i) {
    patReal q = theQuantiles.getQuantile(*i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return result ;
    }
    result.push_back(q) ;
  }
  return result ;

}


void bioSensitivityAnalysis::dumpOnFile(patError*& err) {

  patString fileName = bioParameters::the()->getValueString("OutputFileForSensitivityAnalysis",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  ofstream f(fileName.c_str()) ;
  patULong nValues ;
  for (map<patString,patVariables>::iterator i = theSimulatedValues.begin() ;
       i != theSimulatedValues.end() ;
       ++i) {
    nValues = i->second.size() ;
    f << "\"" << i->first << "\"" << '\t' ;
  }
  f << endl ;
  for (patULong j = 0 ; j < nValues ; ++j) {
    for (map<patString,patVariables>::iterator i = theSimulatedValues.begin() ;
	 i != theSimulatedValues.end() ;
	 ++i) {
      f << (i->second)[j] << '\t' ;
    }
    f << endl ;
  }
  f.close() ;

  patOutputFiles::the()->addUsefulFile(fileName,"Sensitivity analysis");
}
