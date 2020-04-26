//-*-c++-*------------------------------------------------------------
//
// File name : patZhengFosgerau.h
// Author :    \URL[Michel Bierlaire]{http://transp-or2.epfl.ch}
// Date :      Sun Dec 30 08:22:01 2007
//
//--------------------------------------------------------------------

#ifndef patZhengFosgerau_h
#define patZhengFosgerau_h

#include <list>
#include "patOneZhengFosgerau.h"
#include "patSampleEnuGetIndices.h"
#include "patError.h"
#include "patVariables.h"

class patZhengFosgerau {

public:
  patZhengFosgerau(patPythonReal** arrayResult,
		   patULong resRow,
		   patSampleEnuGetIndices* enuIndices,
		   vector<patOneZhengFosgerau>* v,
		   patError*& err) ;
  void compute(patError*& err) ;
  void writeLatexReport(const patString& fileName, patError*& err) ;
  void writeExcelReport(const patString& fileName, patError*& err) ;
private:
  void computeOneTest(patULong zfIndex, patError*& err) ;
  void computeTestForAlt(patVariables* x,
			 patULong zfIndex,
			 patULong alt,
			 patError*& err) ;
  
 private:
  patPythonReal** sampleEnuData ;
  patULong nRows ;
  patSampleEnuGetIndices* sampleEnuIndices ;
  vector<patOneZhengFosgerau>* variablesToTest;
  patULong nAlt ;
  vector<patReal> largest ;
  vector<patReal> smallest ;
  vector<patULong> colIndexOfZfInDataBase ;
  vector<patVariables> zhengTest ;
  vector<patReal> bandwidth ;
  vector<patBoolean> keepRow ;
  vector<vector<patString> > pstricksCode ;
  vector<patString> pstricksDensityCode ;
  vector<vector<patVariables> > xAxis ;
  vector<vector<patVariables> > residuals ;
  vector<vector<patVariables> > lowerResid ;
  vector<vector<patVariables> > upperResid ;
  vector<patString> latexTrimming ;
  vector<patString> textTrimming ;
};

#endif
