//-*-c++-*------------------------------------------------------------
//
// File name : patZhengFosgerau.cc
// Author :    Michel Bierlaire
// Date :      Sun Dec 30 08:36:10 2007
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "patZhengFosgerau.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patOutputFiles.h"
#include "patModelSpec.h"
#include "patZhengTest.h"
#include "patNonParamRegression.h"
#include "patVersion.h"
#include "patPsTricks.h"
#include "patParameters.h"

patZhengFosgerau::patZhengFosgerau(patPythonReal** arrayResult,
				   patULong resRow,
				   patSampleEnuGetIndices* enuIndices,
				   vector<patOneZhengFosgerau>* v,
				   patError*& err):
  sampleEnuData(arrayResult),
  nRows(resRow),
  sampleEnuIndices(enuIndices),
  variablesToTest(v),
  //nAlt nit initialized
  largest(v->size()),
  smallest(v->size()),
  colIndexOfZfInDataBase(v->size(),patBadId),
  zhengTest(v->size()),
  bandwidth(v->size()),
  keepRow(resRow,patFALSE),
  pstricksCode(v->size()),
  pstricksDensityCode(v->size()),
  xAxis(v->size()),
  residuals(v->size()),
  lowerResid(v->size()),
  upperResid(v->size()),
  latexTrimming(v->size()),
  textTrimming(v->size())
{
  
  if (arrayResult == NULL) {
    err = new patErrNullPointer("patReal*") ;
    WARNING(err->describe()) ;
    return ;
  }

  if (sampleEnuIndices == NULL) {
    err = new patErrNullPointer("patSampleEnuGetIndices") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (variablesToTest == NULL) {
    err = new patErrNullPointer("vector<patOneZhengFosgerau>") ;
    WARNING(err->describe()) ;
    return ;

  }

  nAlt = patModelSpec::the()->getNbrAlternatives() ;
}

void patZhengFosgerau::computeOneTest(patULong zfIndex, patError*& err) {
  if (zfIndex >= variablesToTest->size()) {
    err = new patErrOutOfRange<patULong>(zfIndex,0,variablesToTest->size()-1) ;
    WARNING(err->describe()) ;
    return ;
  }

  patOneZhengFosgerau t = (*variablesToTest)[zfIndex] ;
  patULong index = sampleEnuIndices->getIndexZhengFosgerau(&t,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (index == patBadId) {
    stringstream str ;
    str << "Id of variable " << t << " is unknown" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }

  DEBUG_MESSAGE("Index of variables " << zfIndex << " [" << t << "] is " << index) ;
  colIndexOfZfInDataBase[zfIndex] = index ;
  if (index == patBadId) {
    stringstream str ;
    str << "Test not yet implemented for " << t ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }

  // Trim and compute smallest ands largest values

  patULong actualNbrRows(0) ;
  t.resetTrimCounter() ;
  patBoolean first(patTRUE) ;
  for (patULong row = 0 ; row < nRows ; ++row) {
    patReal current = sampleEnuData[row][index] ;
    if (!t.trim(current)) {
      ++actualNbrRows ;
      if (first) {
	largest[zfIndex] = smallest[zfIndex] = current ;
	first = patFALSE ;
      }
      else {
	if (current > largest[zfIndex]) {
	  largest[zfIndex] = current ;
	}
	if (current < smallest[zfIndex]) {
	  smallest[zfIndex] = current ;
	}
      }
      keepRow[row] = patTRUE ;
    }
    else {
      keepRow[row] = patFALSE ;
    }
  }
  GENERAL_MESSAGE("Trimming: " << t.describeTrimming()) ;
  latexTrimming[zfIndex] = t.describeTrimmingLatex() ;
  textTrimming[zfIndex] = t.describeTrimming() ;

  patReal range = largest[zfIndex] - smallest[zfIndex] ;
  
  if (range <= patEPSILON) {
    WARNING("Not enough variation in " << t << ". Test set to zero for all alternatives") ;
    zhengTest[zfIndex] = patVariables(nAlt,0.0) ;
    pstricksCode[zfIndex] = vector<patString>(nAlt,"") ;
    xAxis[zfIndex] = vector<patVariables>(nAlt) ;
    residuals[zfIndex] = vector<patVariables>(nAlt) ;
    lowerResid[zfIndex] = vector<patVariables>(nAlt) ;
    upperResid[zfIndex] = vector<patVariables>(nAlt) ;
    return ;
  }
  
  // Populate the vectors
  bandwidth[zfIndex] = t.bandwidth / sqrt(patReal(actualNbrRows)) ;

  patVariables normalizedVariableToTest ;
  //  patVariables variableToTest ;
  for (patULong row = 0 ; row < nRows ; ++row) {
    if (keepRow[row]) {
      patReal current = (sampleEnuData[row][index] - smallest[zfIndex]) / range ;
      normalizedVariableToTest.push_back(current) ;
      //      variableToTest.push_back(sampleEnuData[row][index]) ;
    }
  }
  zhengTest[zfIndex].resize(nAlt) ;
  pstricksCode[zfIndex].resize(nAlt) ;
  xAxis[zfIndex].resize(nAlt) ;
  residuals[zfIndex].resize(nAlt) ;
  upperResid[zfIndex].resize(nAlt) ;
  lowerResid[zfIndex].resize(nAlt) ;
  for(unsigned long alt = 0 ; alt < nAlt ; ++alt) {
    computeTestForAlt(&normalizedVariableToTest,zfIndex,alt,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}

void patZhengFosgerau::computeTestForAlt(patVariables* x,
					 patULong zfIndex,
					 patULong alt,
					 patError*& err) {


  patOneZhengFosgerau t = (*variablesToTest)[zfIndex] ;

  unsigned long indexResid = sampleEnuIndices->getIndexResid(alt,err) ;
  if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
  }
  
  unsigned long indexProba = sampleEnuIndices->getIndexProba(alt,err) ;
  if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
  }
  
  patULong indexT = colIndexOfZfInDataBase[zfIndex] ;
  if (indexT == patBadId) {
    stringstream str ;
    str << "Index unknown for variables " << zfIndex << " [" << t << "]" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }
  patVariables residual ;
  patVariables probabilities ;
  // Populate the vectors
  for (patULong row = 0 ; row < nRows ; ++row) {
    if (keepRow[row]) {
      residual.push_back(sampleEnuData[row][indexResid]) ;
      probabilities.push_back(sampleEnuData[row][indexProba]) ;
    }
  }

  // Compute the test
  
  patZhengTest theTest(x,&residual,bandwidth[zfIndex],err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  if (zfIndex >= zhengTest.size()) {
    err = new patErrOutOfRange<patULong>(zfIndex,0,zhengTest.size()-1) ;
    WARNING(err->describe()) ;
    return ;
  }
  
  if (alt >= zhengTest[zfIndex].size()) {
    err = new patErrOutOfRange<patULong>(alt,0,zhengTest[zfIndex].size()-1) ;
    WARNING(err->describe()) ;
    return ;
    
  }
  
  zhengTest[zfIndex][alt] = theTest.compute(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  // Prepare the plot
  
  patNonParamRegression npReg(&residual, 
			      x, 
			      &probabilities,
			      bandwidth[zfIndex],
			      patParameters::the()->getgevNonParamPlotRes(),
			      err) ;
  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  npReg.compute(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patVariables* nx = npReg.getNewX() ;
  patVariables rescaleX(nx->size()) ;
  for (patULong ii = 0 ; ii < nx->size() ; ++ii) {
    rescaleX[ii] = smallest[zfIndex] + (*nx)[ii] * (largest[zfIndex]-smallest[zfIndex]) ;
  }

  xAxis[zfIndex][alt] = rescaleX ;
  residuals[zfIndex][alt] = *npReg.getMainPlot() ;
  if (npReg.getLowerPlot() != NULL) {
    lowerResid[zfIndex][alt] = *npReg.getLowerPlot() ;
  }
  if (npReg.getUpperPlot() != NULL) {
    upperResid[zfIndex][alt] = *npReg.getUpperPlot() ;
  }

  patPsTricks thePst(&rescaleX,
		     npReg.getMainPlot(),
		     npReg.getUpperPlot(),
		     npReg.getLowerPlot(),
		     err) ;
	  
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }


  if (zfIndex >= pstricksCode.size()) {
    err = new patErrOutOfRange<patULong>(zfIndex,0,pstricksCode.size()-1) ;
    WARNING(err->describe()) ;
    return ;
  }
  
  if (alt >= pstricksCode[zfIndex].size()) {
    err = new patErrOutOfRange<patULong>(alt,0,pstricksCode[zfIndex].size()-1) ;
    WARNING(err->describe()) ;
    return ;
    
  }
  

  pstricksCode[zfIndex][alt] = 
    thePst.getCode(patParameters::the()->getgevNonParamPlotMaxY() ,
		   patParameters::the()->getgevNonParamPlotXSizeCm(),
		   patParameters::the()->getgevNonParamPlotYSizeCm() ,
		   patParameters::the()->getgevNonParamPlotMinXSizeCm(),
		   patParameters::the()->getgevNonParamPlotMinYSizeCm() ,
		   err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patVariables rescaledDensity(*npReg.getDensityPlot()) ;
  rescaledDensity /= (largest[zfIndex]-smallest[zfIndex]) ;
  patPsTricks theDensityPst(&rescaleX,
			    &rescaledDensity,
			    err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  pstricksDensityCode[zfIndex] = 
    theDensityPst.getCode(patParameters::the()->getgevNonParamPlotMaxY() ,
			  patParameters::the()->getgevNonParamPlotXSizeCm(),
			  patParameters::the()->getgevNonParamPlotYSizeCm() ,
			  patParameters::the()->getgevNonParamPlotMinXSizeCm(),
			  patParameters::the()->getgevNonParamPlotMinYSizeCm() ,
			  err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

}

void patZhengFosgerau::compute(patError*& err) {
  for (patULong i = 0 ; i < variablesToTest->size() ; ++i) {
    computeOneTest(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }
}

void patZhengFosgerau::writeLatexReport(const patString& fileName, patError*& err) {
  DEBUG_MESSAGE("Write LaTeX file " << fileName) ;
  ofstream latexFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;
  
  latexFile << "\\documentclass[12pt]{article}" << endl ;
  latexFile << "" << endl ;

  latexFile << "\\usepackage{pstricks}" << endl ;
  latexFile << "\\usepackage{pst-plot}" << endl ;

  latexFile << "\\title{BIOSIM: Zheng-Fosgerau test \\\\" ;
  latexFile << "{\\footnotesize "<< patVersion::the()->getVersionInfo()<<" \\\\  Compiled "<<patVersion::the()->getVersionDate()<<"}}" << endl ;
  latexFile << "\\author{"<<patVersion::the()->getVersionInfoAuthor()<<"}" << endl ;
  latexFile << "\\date{Report created on "<<now.getTimeString(patTsfFULL)<<"}" << endl ;
  
  
  latexFile << "\\begin{document}" << endl ;
  latexFile << "" << endl ;
  latexFile << "\\maketitle" << endl ;

  patFormatRealNumbers theNumber;
  for (patULong i = 0 ; i < variablesToTest->size() ; ++i) {
    patOneZhengFosgerau t = (*variablesToTest)[i] ;
    latexFile << "\\section{Testing " << t.latexDescription() << "}" << endl ; 

    latexFile << "\\begin{center}" << endl ;
    latexFile << "\\begin{tabular}{rll}" << endl ;
    latexFile << "Smallest value: & " << theNumber.format(patFALSE,
							  patTRUE,
							  3,
							  smallest[i]) << "\\\\" << endl ;
    latexFile << "Largest value: & " << theNumber.format(patFALSE,
							 patTRUE,
							 3,
							 largest[i]) << "\\\\" << endl ;
    latexFile << "Range: & " << theNumber.format(patFALSE,
						 patTRUE,
						 3,
						 largest[i]-smallest[i]) << "\\\\" << endl ;
    latexFile << "Bandwidth: & " << theNumber.format(patFALSE,
						     patTRUE,
						     3,
						     bandwidth[i]) << "\\\\" << endl ;
    latexFile << "Trimming: & \\multicolumn{2}{l}{" << latexTrimming[i] << "}\\\\" << endl ;

    latexFile << "\\hline" << endl ;
    latexFile << "\\multicolumn{3}{l}{Value of the test for each alternative} \\\\" << endl ;
    latexFile << "\\hline" << endl ;
    for (patULong alt = 0 ; alt < nAlt ; ++alt) {
      patULong userId = patModelSpec::the()->getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patString name = patModelSpec::the()->getAltName(userId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      latexFile << name << " (" << userId << "): & " ;
      latexFile << theNumber.format(patFALSE,patTRUE,3,zhengTest[i][alt]) ;
      latexFile << " & " ;
      if (zhengTest[i][alt] > 1.645) {
	latexFile << " \\textbf{rejected} at 5\\% level" << endl ;
      }
      latexFile << "\\\\" << endl ;
    }
    latexFile << "\\hline" << endl ;
    
    latexFile << "\\end{tabular}" << endl ;
    latexFile << "\\end{center}" << endl ;
    latexFile << "\\begin{center}" << endl ;
    
    
    latexFile << "\\begin{tabular}{rl}" << endl ;
    latexFile << "\\multicolumn{2}{l}{Residual analysis}\\\\" << endl ;
    latexFile << "\\hline" << endl ;
    stringstream astr ; 
    astr << "fig:" << i << "_proba"  ;
    latexFile << "Probability: & Figure \\ref{" << astr.str() << "} \\\\" << endl ;
    for (patULong alt = 0 ; alt < nAlt ; ++alt) {
      patULong userId = patModelSpec::the()->getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patString name = patModelSpec::the()->getAltName(userId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      stringstream str ; 
      str << "fig:" << i << "_" << alt ;
      latexFile << name
		<< " (" 
		<< userId 
		<< "): & Figure \\ref{" 
		<< str.str() 
		<< "}\\\\" 
		<< endl ;
      
    }
    latexFile << "\\end{tabular}" << endl ;
    latexFile << "\\end{center}" << endl ;

    latexFile << "\\begin{figure}" << endl ;
    latexFile << "\\begin{center}" << endl ;
    latexFile << pstricksDensityCode[i] << endl;
      latexFile << "\\caption{\\label{" << astr.str() << "}Density of " << t.latexDescription() << "}" << endl ;
      latexFile << "\\end{center}" << endl ;
      latexFile << " \\end{figure}" << endl ;

    for (patULong alt = 0 ; alt < nAlt ; ++alt) {
      patULong userId = patModelSpec::the()->getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patString name = patModelSpec::the()->getAltName(userId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      stringstream str ; 
      str << "fig:" << i << "_" << alt ;
      
      latexFile << "\\begin{figure}" << endl ;
      latexFile << "\\begin{center}" << endl ;
      latexFile << pstricksCode[i][alt] << endl;
      latexFile << "\\caption{\\label{" << str.str() << "}Testing " << t.latexDescription() << " with alt. " << name << "}" << endl ;
      latexFile << "\\end{center}" << endl ;
      latexFile << " \\end{figure}" << endl ;
    }
  

    latexFile << "\\clearpage" << endl ;
  }  


  latexFile << "\\end{document}" << endl ;
  latexFile.close() ;
  patOutputFiles::the()->addCriticalFile(fileName,"Results of the Zheng-Fosgerau test in LaTeX format");
  return ;

}

void patZhengFosgerau::writeExcelReport(const patString& fileName, patError*& err) {
  DEBUG_MESSAGE("Write file " << fileName) ;
  ofstream file(fileName.c_str()) ;
  patAbsTime now ;
  patFormatRealNumbers theNumber;
  for (patULong i = 0 ; i < variablesToTest->size() ; ++i) {
    patOneZhengFosgerau t = (*variablesToTest)[i] ;
    file << '\"' << t.latexDescription() << '\"' << '\t' ;
    file << "\"Smallest value:\"\t\t" << theNumber.format(patFALSE,
							patTRUE,
							3,
							smallest[i]) << endl ;
    file << '\"' << t.latexDescription() << '\"' << '\t' ;
    file << "\"Largest value:\"\t\t" << theNumber.format(patFALSE,
						       patTRUE,
						       3,
						       largest[i]) << endl ;
    file << '\"' << t.latexDescription() << '\"' << '\t' ;
    file << "\"Range:\"\t\t" << theNumber.format(patFALSE,
						patTRUE,
						3,
						largest[i]-smallest[i]) << endl ;
    file << '\"' << t.latexDescription() << '\"' << '\t' ;
    file << "\"Bandwidth:\"\t\t" << theNumber.format(patFALSE,
						   patTRUE,
						   3,
						   bandwidth[i]) << endl ;
    file << '\"' << t.latexDescription() << '\"' << '\t' ;
    file << "\"Trimming:\"\t\t\"" << textTrimming[i] << "\"" << endl ;

    for (patULong alt = 0 ; alt < nAlt ; ++alt) {
      patULong userId = patModelSpec::the()->getAltId(alt,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patString name = patModelSpec::the()->getAltName(userId,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      file << '\"' << t.latexDescription() << '\"' << '\t' ;
      file << name << '\t' << userId  << '\t' ;
      file << theNumber.format(patFALSE,patTRUE,3,zhengTest[i][alt]) ;
      file << endl ;
      file << '\"' << t.latexDescription() << '\"' << '\t' ;
      file << "\"x-axis:\"" << '\t' ;
      for (patULong ii = 0 ; ii < xAxis[i][alt].size() ; ++ii) {
	file << xAxis[i][alt][ii] << '\t' ;
      }
      file << endl ;
      file << '\"' << t.latexDescription() << '\"' << '\t' ;
      file << "\"Residuals:\"" << '\t' ;
      for (patULong ii = 0 ; ii < residuals[i][alt].size() ; ++ii) {
	file << residuals[i][alt][ii] << '\t' ;
      }
      file << endl ;
      file << '\"' << t.latexDescription() << '\"' << '\t' ;
      file << "\"Lower:\"" << '\t' ;
      for (patULong ii = 0 ; ii < lowerResid[i][alt].size() ; ++ii) {
	file << lowerResid[i][alt][ii] << '\t' ;
      }
      file << endl ;
      file << '\"' << t.latexDescription() << '\"' << '\t' ;
      file << "\"Upper:\"" << '\t' ;
      for (patULong ii = 0 ; ii < upperResid[i][alt].size() ; ++ii) {
	file << upperResid[i][alt][ii] << '\t' ;
      }
      file << endl ;
    }  
    file << "\"-----------------------------------------\"" <<  endl ;
    file << endl ;
  }

  file << endl ;
  file.close() ;
  return ;

}
