//-*-c++-*------------------------------------------------------------
//
// File name : bioSensitivityAnalysis.h
// Author :    Michel Bierlaire
// Date :      Sun May  6 09:28:34 2012
//
//--------------------------------------------------------------------

#ifndef bioSensitivityAnalysis_h
#define bioSensitivityAnalysis_h

#include <map>
#include "patError.h"
#include "patVariables.h"
#include "patString.h"

class bioSensitivityAnalysis {
 public:
  bioSensitivityAnalysis() ;
  void addData(patString name, patReal value) ;
  vector<patReal> getQuantiles(patString name, 
			       vector<patReal> alphas, 
			       patError*& err) ;
  void dumpOnFile(patError*& err) ;
 private:
  map<patString,patVariables> theSimulatedValues ;
  
};

#endif
