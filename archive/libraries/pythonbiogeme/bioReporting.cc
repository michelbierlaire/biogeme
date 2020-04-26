//-*-c++-*------------------------------------------------------------
//
// File name : bioReporting.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Mon Aug 17 18:56:15 2009
//
//--------------------------------------------------------------------

#include <set>
#include <iomanip>

#include <unistd.h>
#include "patCompareCorrelation.h"
#include "patCorrelation.h"
#include "patOutputFiles.h"
#include "patErrNullPointer.h"
#include "patErrOutOfRange.h"
#include "patMath.h"
#include "bioReporting.h"
#include "bioRawResults.h"
#include "patDisplay.h"
#include "patSvd.h"
#include "patPValue.h"
#include "bioLiteralRepository.h"
#include "bioVersion.h"
#include "bioParameters.h"
#include "patFormatRealNumbers.h"
#include "bioSample.h"
#include "bioRandomDraws.h"
#include "bioBayesianResults.h"
#include "patStatistics.h"
#include "patKalman.h"
#include "bioSimulatedValues.h"

bioReporting::bioReporting(patError*& err) :
  reportDraws(patFALSE),
  theRawResults(NULL),
  theBayesianResults(NULL),
  nbrDraws(bioRandomDraws::the()->nbrOfDraws()),
  nbrParameters(patBadId),
  sampleSize(patBadId),
  removedObservations(patBadId),
  initLoglikelihood(0.0),
  algorithm(""),
  diagnostic(""),
  iterations(patBadId),
  runTime(""),
  dataProcessingTime(""),
  sampleFile(""),
  simulatedValues(NULL),
  parameterRead(patFALSE),
  sensitivityAnalysis(patFALSE) {

}

void bioReporting::computeEstimationResults(bioRawResults* rr, patError*& err) {
  theRawResults = rr ;
  if (rr == NULL) {
    err = new patErrNullPointer("bioRawResults") ;
    WARNING(err->describe()) ;
    return ;
  }
  compute(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
}

void bioReporting::compute(patError*& err) {

  // Set the beta values
  bioLiteralRepository::the()->setBetaValues(theRawResults->solution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  // Standard errors

  if (theRawResults->varianceCovariance != NULL) {
    for (patULong i = 0 ; i < theRawResults->varianceCovariance->nRows() ; ++i) {
      if ((*theRawResults->varianceCovariance)[i][i] <= patEPSILON) {
	stdErr.push_back(patMaxReal) ;
      }
      else{
	stdErr.push_back(sqrt((*theRawResults->varianceCovariance)[i][i])) ; 
      }
    }
  }

  
  if (theRawResults->robustVarianceCovariance != NULL) {
    for (patULong i = 0 ; i < theRawResults->robustVarianceCovariance->nRows() ; ++i) {
      if ((*theRawResults->robustVarianceCovariance)[i][i] <= patEPSILON) {
	robustStdErr.push_back(patMaxReal) ;
      } 
      else {
	robustStdErr.push_back(sqrt((*theRawResults->robustVarianceCovariance)[i][i])) ; 
      }
    }
  }

  nbrParameters = bioLiteralRepository::the()->getNumberOfEstimatedParameters() ;
  
}

patBoolean bioReporting::isRaoCramerAvailable() const {
  if (theRawResults == NULL) {
    return patFALSE ;
  }
  return (theRawResults->varianceCovariance != NULL) ;
}

patBoolean bioReporting::isSandwichAvailable() const {
  if (theRawResults == NULL) {
    return patFALSE ;
  }
  return (theRawResults->robustVarianceCovariance != NULL) ;
}


void bioReporting::writeLaTeX(patString fileName, patError*& err) {

  ofstream latexFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;

  latexFile << "%% This file is designed to be included into a LaTeX document" << endl ;
  latexFile << "%% See http://www.latex-project.org/ for information about LaTeX" << endl ;
  latexFile << "%% Biogeme " << bioVersion::the()->getVersion() << endl ;
  latexFile << "%% Compiled " << bioVersion::the()->getVersionInfoDate() << endl ;
  latexFile << "%% " << bioVersion::the()->getVersionInfoAuthor()  << endl ;
  latexFile << "%% Report created on " << now.getTimeString(patTsfFULL) << endl ;

  latexFile << "  \\begin{tabular}{l}" << endl ;

  latexFile << "\\begin{tabular}{rlr@{.}lr@{.}lr@{.}lr@{.}l}" << endl ;
  if (isSandwichAvailable()) {
    latexFile << "         &                       &   \\multicolumn{2}{l}{}    & \\multicolumn{2}{l}{Robust}  &     \\multicolumn{4}{l}{}   \\\\" << endl ;
  }
  latexFile << "Parameter &                       &   \\multicolumn{2}{l}{Coeff.}      & \\multicolumn{2}{l}{Asympt.}  &     \\multicolumn{4}{l}{}   \\\\" << endl ;
  latexFile << "number &  Description                     &   \\multicolumn{2}{l}{estimate}      & \\multicolumn{2}{l}{std. error}  &   \\multicolumn{2}{l}{$t$-stat}  &   \\multicolumn{2}{l}{$p$-value}   \\\\" << endl ;
  latexFile << "" << endl ;
  latexFile << "\\hline" << endl ;
  latexFile << "" << endl ;
  
  patIterator<bioFixedParameter*>* theBetaIter = bioLiteralRepository::the()->getSortedIteratorFixedParameters() ;


 patULong paramNumber = 0 ;
  for (theBetaIter->first() ;
       !theBetaIter->isDone() ;
       theBetaIter->next()) { // loop on beta  parameters
    bioFixedParameter* beta = theBetaIter->currentItem() ;
    if (!beta->isFixedParam()) {
      ++paramNumber ;
      latexFile << paramNumber << " & " ; 
      patULong i = beta->getEstimatedParameterId() ;
      patReal s ;
      if (isSandwichAvailable()) {
	if (i >= robustStdErr.size()) {
	  err = new patErrOutOfRange<patULong>(i,0,robustStdErr.size()-1) ;
	  WARNING(err->describe()) ;
	  return ;
	}
	s = robustStdErr[i] ;
      }
      else {
	s = stdErr[i] ;
	if (i >= stdErr.size()) {
	  err = new patErrOutOfRange<patULong>(i,0,robustStdErr.size()-1) ;
	  WARNING(err->describe()) ;
	  return ;
	}
      }
      latexFile << beta->getLaTeXRow(s,err) << "\\\\" << endl;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
  }
  DELETE_PTR(theBetaIter) ;
  latexFile << "\\hline" << endl ;
  latexFile << "\\end{tabular}" << endl ;

 latexFile << "\\\\" << endl ; 
  patFormatRealNumbers theNumber ;

  latexFile << "\\begin{tabular}{rcl}" << endl ;
  latexFile << "\\multicolumn{3}{l}{\\bf Summary statistics}\\\\" << endl ;
  latexFile << "\\multicolumn{3}{l}{ Number of observations = $"<< sampleSize <<"$} \\\\" << endl ;
  latexFile << "\\multicolumn{3}{l}{ Number of excluded observations = $"<< removedObservations <<"$} \\\\" << endl ;
  latexFile << "\\multicolumn{3}{l}{ Number of estimated  parameters = $"<< nbrParameters <<"$} \\\\" << endl ;
  if (initLoglikelihood != 0.0) {
    latexFile << " $\\mathcal{L}(\\beta_0)$ &=&  $"<< theNumber.formatStats(initLoglikelihood)<<"$ \\\\" << endl ;
  }
  if (theRawResults != NULL) {
    latexFile << " $\\mathcal{L}(\\hat{\\beta})$ &=& $"<<theNumber.formatStats(theRawResults->valueAtSolution)<<" $  \\\\" << endl ;
    if (initLoglikelihood != 0.0) {
      latexFile << " $-2[\\mathcal{L}(\\beta_0) -\\mathcal{L}(\\hat{\\beta})]$ &=& $"<<theNumber.formatStats( -2.0 * (initLoglikelihood - theRawResults->valueAtSolution))<<"$ \\\\" << endl ;
      latexFile << "    $\\rho^2$ &=&   $"<<theNumber.formatStats(1.0 - (theRawResults->valueAtSolution / initLoglikelihood))<<"$ \\\\" << endl ;
      latexFile << "    $\\bar{\\rho}^2$ &=&    $"<< theNumber.formatStats(1.0 - ((theRawResults->valueAtSolution-nbrParameters) / initLoglikelihood))<<"$ \\\\" << endl ;
    }
  }

  latexFile << "\\end{tabular}" << endl ;


  latexFile << "  \\end{tabular}" << endl ;

  latexFile.close() ;
  patOutputFiles::the()->addUsefulFile(fileName,"Results in LaTeX format");

}

void bioReporting::writeHtml(patString fileName, patError*& err) {
  patFormatRealNumbers theNumber ;
  ofstream htmlFile(fileName.c_str()) ;
  patAbsTime now ;
  now.setTimeOfDay() ;

  htmlFile << "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">" << endl ;
  htmlFile << "" << endl ;
  htmlFile << "<html>" << endl ;
  htmlFile << "<head>" << endl ;
  htmlFile << "<script src=\"http://transp-or.epfl.ch/biogeme/sorttable.js\"></script>" << endl ;
  htmlFile << "<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />" << endl ;
  htmlFile << "<title>"<< fileName <<" - Report from " << bioVersion::the()->getVersionInfoDate() << "</title>" << endl ;
  htmlFile << "<meta name=\"keywords\" content=\"biogeme, discrete choice, random utility\">" << endl ;
  htmlFile << "<meta name=\"description\" content=\"Report from " << bioVersion::the()->getVersionInfoDate() << "\">" << endl ;
  htmlFile << "<meta name=\"author\" content=\"Michel Bierlaire\">" << endl ;
  htmlFile << "<style type=text/css>" << endl ;
  htmlFile << "<!--table" << endl ;
  htmlFile << ".biostyle" << endl ;
  htmlFile << "	{font-size:10.0pt;" << endl ;
  htmlFile << "	font-weight:400;" << endl ;
  htmlFile << "	font-style:normal;" << endl ;
  htmlFile << "	font-family:Courier;}" << endl ;
  htmlFile << ".boundstyle" << endl ;
  htmlFile << "	{font-size:10.0pt;" << endl ;
  htmlFile << "	font-weight:400;" << endl ;
  htmlFile << "	font-style:normal;" << endl ;
  htmlFile << "	font-family:Courier;" << endl ;
  htmlFile << "        color:red}" << endl ;
  htmlFile << "-->" << endl ;
  htmlFile << "</style>" << endl ;
  htmlFile << "</head>" << endl ;
  htmlFile << "" << endl ;
  htmlFile << "<body bgcolor=\"#ffffff\">" << endl ;

  htmlFile << "<p>" << bioVersion::the()->getVersionInfoDate() << "</p>" << endl ;
  htmlFile << "<p>Home page: <a href='" << bioVersion::the()->getUrl() << "' target='_blank'>"<<  bioVersion::the()->getUrl() <<"</a></p>" << endl ;
  htmlFile << "<p>Submit questions to <a href='" << bioVersion::the()->getUrlUsersGroup() << "' target='_blank'>"<<  bioVersion::the()->getUrlUsersGroup() <<"</a></p>" << endl ;

  htmlFile << "<p>" << bioVersion::the()->getVersionInfoAuthor() << "</p>" << endl ;
  htmlFile << "<p>This file has automatically been generated on" << endl ;
  htmlFile <<  now.getTimeString(patTsfFULL) << "</p>" << endl ;
  htmlFile << "<p>If you drag this HTML file into the Calc application of <a href='http://www.openoffice.org/' target='_blank'>OpenOffice</a>, or the spreadsheet of <a href='https://www.libreoffice.org/' target='_blank'>LibreOffice</a>, you will be able to perform additional calculations.</p>" ;

  htmlFile << "<table>" ;
    htmlFile << "<tr class=biostyle><td align=right><strong>Report file</strong>:	</td>" ;
    htmlFile << "<td>" << fileName  << "</td></tr>" ;
  if (sampleFile != "") {
    htmlFile << "<tr class=biostyle><td align=right><strong>Sample file</strong>:	</td>" ;
    htmlFile << "<td>" << sampleFile  << "</td></tr>" ;
  }
  
  htmlFile << "</table>" ;

  if (!formulas.empty()) {
    htmlFile << "<h1>Formulas</h1>" << endl ;
    htmlFile << "<table border=\"0\">" << endl ;
    for (map<patString, patString>::iterator s = formulas.begin() ;
	 s != formulas.end() ;
	 ++s) {
      htmlFile << "<tr class=biostyle><td align=right><strong><em>"<<s->first<<":</em></strong>	</td>" ;
      htmlFile << "<td><em>" << s->second  << "</em></td></tr>" ;
      
    }
    htmlFile << "</table>" << endl ;
  }

  if (!statistics.empty()) {
    htmlFile << "<h1>Statistics</h1>" << endl ;
    htmlFile << "<table border=\"0\">" << endl ;
    for (map<patString, patReal>::iterator s = statistics.begin() ;
	 s != statistics.end() ;
	 ++s) {
      htmlFile << "<tr class=biostyle><td align=right><strong><em>"<<s->first<<":</em></strong>	</td>" ;
      htmlFile << "<td><em>" << s->second  << "</em></td></tr>" ;
      
    }
    htmlFile << "</table>" << endl ;
  }


  if (theRawResults != NULL) {
    
    htmlFile << "<h1>Estimation report</h1>" << endl ;
    htmlFile << endl ;
    
    
    htmlFile << "<table border=\"0\">" << endl ;
    
    nbrDraws = bioRandomDraws::the()->nbrOfDraws() ;

    
    if (reportDraws && nbrDraws != patBadId && nbrDraws != 0) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Number of draws</strong>:</td> <td>"<< nbrDraws <<"</td></tr>" << endl ;
    }
    if (nbrParameters != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Number of estimated parameters</strong>: </td> <td>"<< nbrParameters <<"</td></tr>" << endl ;
    }
    if (sampleSize != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Sample size</strong>:</td> <td>"<< sampleSize <<"</td></tr>" << endl ;
    }
    
    if (removedObservations != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Excluded observations</strong>:</td> <td>"<< removedObservations <<"</td></tr>" << endl ;

    }
    if (initLoglikelihood != 0.0) {
      htmlFile << "<tr class=biostyle><td align=right><strong>Init log likelihood</strong>:	</td> <td>"<< theNumber.formatStats(initLoglikelihood) 
	       <<"</td></tr>" << endl ;
    }
    if (theRawResults != NULL) {
      htmlFile << "<tr class=biostyle><td align=right><strong>Final log likelihood</strong>:	</td> <td>"
	       << theNumber.formatStats(theRawResults->valueAtSolution) 
	       <<"</td></tr>" << endl ;
      htmlFile << "<tr class=biostyle><td align=right><strong>Likelihood ratio test for the init. model</strong>:	</td> <td>"
	       << theNumber.formatStats(-2.0 * (initLoglikelihood  - theRawResults->valueAtSolution))  
	       <<"</td></tr>" << endl ;
      htmlFile << "<tr class=biostyle><td align=right><strong>Rho-square for the init. model</strong>:	</td> <td>"
	       << theNumber.formatStats(1.0-(theRawResults->valueAtSolution/initLoglikelihood))
	       <<"</td></tr>" << endl ;
      htmlFile << "<tr class=biostyle><td align=right><strong>Rho-square-bar for the init. model</strong>:	</td> <td>"
	       << theNumber.formatStats(1.0-((theRawResults->valueAtSolution-patReal(nbrParameters))/initLoglikelihood)) 
	       <<"</td></tr>" << endl ;

            htmlFile << "<tr class=biostyle><td align=right><strong>Akaike Information Criterion</strong>:	</td> <td>"
		     << theNumber.formatStats(2.0 * patReal(nbrParameters) - 2.0 * theRawResults->valueAtSolution) 
	       <<"</td></tr>" << endl ;

	    htmlFile << "<tr class=biostyle><td align=right><strong>Bayesian Information Criterion</strong>:	</td> <td>"
		     << theNumber.formatStats(- 2.0 * theRawResults->valueAtSolution + patReal(nbrParameters) * log(patReal(sampleSize)) ) 
	       <<"</td></tr>" << endl ;

      
      htmlFile << "<tr class=biostyle><td align=right><strong>Final gradient norm</strong>:	</td> <td>"
	       << theNumber.format(patTRUE,
				   patFALSE,
				   3,
				   theRawResults->gradientNorm) 
	       <<"</td></tr>" << endl ;
      htmlFile << "<tr class=biostyle><td align=right><strong>Diagnostic</strong>:	</td> <td>" ;
      if (algorithm != "") {
	htmlFile << algorithm << ": " ;
      }
      htmlFile << diagnostic 
	       <<"</td></tr>" << endl ;
    }
    if (iterations != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right><strong>Iterations</strong>:	</td> <td>"
	       << iterations
	       <<"</td></tr>" << endl ;
    }
    if (dataProcessingTime != "") {
      htmlFile << "<tr class=biostyle><td align=right><strong>Data processing time</strong>:	</td> <td>"
	       << dataProcessingTime
	       <<"</td></tr>" << endl ;
    }
    if (runTime != "") {
      htmlFile << "<tr class=biostyle><td align=right><strong>Run time</strong>:	</td> <td>"
	       << runTime
	       <<"</td></tr>" << endl ;
    }
    int nThreads = bioParameters::the()->getValueInt("numberOfThreads") ;
    htmlFile << "<tr class=biostyle><td align=right><strong>Nbr of threads</strong>:	</td> <td>"
	     << nThreads
	     <<"</td></tr>" << endl ;
  

    htmlFile << "</table>" << endl ;

    patIterator<bioFixedParameter*>* theBetaIter = bioLiteralRepository::the()->getSortedIteratorFixedParameters() ;
    patIterator<bioFixedParameter*>* theSecondBetaIter = bioLiteralRepository::the()->getSortedIteratorFixedParameters() ;

    if (theRawResults != NULL) {
      htmlFile << "<h2>Estimated parameters</h2>" << endl ;
      htmlFile << "<p><font size='-1'>Click on the headers of the columns to sort the table  [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle>" ;
      // Col 1
      htmlFile << "<th>Name </th>" ;
      // Col 2
      htmlFile << "<th>Value</th>" ;
      // Col 3
      htmlFile << "<th>Std err</th>" ;
      // Col 4
      htmlFile << "<th>t-test</th>" ;
      int printPValue = bioParameters::the()->getValueInt("printPValue",err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (printPValue != 0) {
	// Col 5
	htmlFile << "<th>p-value</th>" ;
      }
      // Col 6
      htmlFile << "<th></th>" ;
      if (isSandwichAvailable()) {
	// Col 7
	htmlFile << "<th>Robust Std err</th>" ;
	// Col 8
	htmlFile << "<th>Robust t-test</th>" ;
	if (printPValue) {
	  // Col 9
	  htmlFile << "<th>p-value</th>" ;
	}
	// Col 10
	htmlFile << "<th></th>" ;
      }
      htmlFile << "</tr>" ;
      htmlFile << endl ;
    
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    
      patReal tTestThreshold = bioParameters::the()->getValueReal("tTestThreshold",err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patString warningSign = bioParameters::the()->getValueString("warningSign",err) ; 
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    

      for (theBetaIter->first() ;
	   !theBetaIter->isDone() ;
	   theBetaIter->next()) { // loop on beta  parameters

	bioFixedParameter* beta = theBetaIter->currentItem() ;

	if (beta == NULL) {
	  err = new patErrNullPointer("bioFixedParameter") ;
	  WARNING(err->describe()) ;
	  return ;
	}

      
	if (!beta->isFixedParam()) {
	

	  htmlFile << "<!-- " << *beta << "-->" ;
	  patULong i = beta->getEstimatedParameterId() ;
	
	  patReal theValue = beta->getValue() ;
	
	  if (patAbs(theValue - beta->getLowerBound()) <= patEPSILON ||
	      patAbs(theValue - beta->getUpperBound()) <= patEPSILON) {
	    htmlFile << "<tr class=boundstyle>" << endl ;
	  }
	  else {
	    htmlFile << "<tr class=biostyle>" << endl ;
	  }
	  // Col 1 
	  htmlFile << "<td>" << beta->getName() << "</td>" ;
	  // Col 2
	  htmlFile << "<td>" << theNumber.formatParameters(theValue) << "</td>" ;
	
	  if (isRaoCramerAvailable()) {
	    
	    patReal ttest ;
	    if (stdErr[i] != patMaxReal) {
	      ttest = theValue / stdErr[i] ;
	    }
	    else {
	      ttest = 0.0;
	    }
	    // Col 3
	    htmlFile << "<td>" << theNumber.formatParameters(stdErr[i]) << "</td>";
	    // Col 4
	    htmlFile << "<td>" << theNumber.formatTTests(ttest) << "</td>" ;
	    if (printPValue) {
	      patReal pvalue = patPValue(patAbs(ttest),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      // Col 5
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    }
	    // Col 6
	    if (patAbs(ttest) < tTestThreshold || !patFinite(ttest)) {
	      htmlFile << "<td>" << warningSign << "</td>";
	    }
	    else {
	      htmlFile << "<td></td>" ;
	    }
	  }
	  else {
	    // Col 3
	    htmlFile << "<td>var-covar unavailable</td>" ;
	  }
	  if (isSandwichAvailable()) {
	    patReal rttest;
	    if (robustStdErr[i] != patMaxReal) {
	      rttest = theValue  / robustStdErr[i] ; 
	    }
	    else {
	      rttest = 0.0 ;
	    }
	    // Col 7
	    htmlFile << "<td>" << theNumber.formatParameters(robustStdErr[i]) << "</td>" ;
	    // Col 8
	    htmlFile << "<td>" << theNumber.formatTTests(rttest) << "</td>" ; 
	    if (printPValue) {
	    
	      patReal pvalue = patPValue(patAbs(rttest),err) ;
	      if (err != NULL) {
		WARNING(err->describe()) ;
		return ;
	      }
	      // Col 9
	      htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	    }
	    // Col 10
	    if (patAbs(rttest) < tTestThreshold ||
		!patFinite(rttest)) {
	      htmlFile << "<td>" << warningSign << "</td>";
	    }
	    else {
	      htmlFile << "<td></td>" ;
	    }
	  }
	  else {
	    // Col 7
	    htmlFile << "<td></td>" ;
	    // Col 8
	    if (printPValue) {
	      htmlFile << "<td></td>" ;
	    }
	    // Col 9
	    htmlFile << "<td></td>" ;
	    // Col 10
	    htmlFile << "<td></td>" ;
	  }
	  htmlFile << "</tr>" << endl ;
	}
    
      }

      htmlFile << "</table>" << endl ;
      
      std::set<patCorrelation,patCompareCorrelation> correlation ;
      for (theBetaIter->first() ;
	   !theBetaIter->isDone() ;
	   theBetaIter->next()) { // loop on beta  parameters
      


	bioFixedParameter* betai = theBetaIter->currentItem() ;
	if (!betai->isFixedParam()) {
	  patULong i = betai->getEstimatedParameterId() ;
	  if (betai == NULL) {
	    err = new patErrNullPointer("bioFixedParameter") ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  patReal varI(0) ;
	  if (isRaoCramerAvailable()) {
	    varI = (*theRawResults->varianceCovariance)[i][i] ;
	  }
	  patReal robustVarI(0) ;
	  if (isSandwichAvailable()) {
	    robustVarI = (*theRawResults->robustVarianceCovariance)[i][i] ;
	  }    
	  for (theSecondBetaIter->first() ;
	       !theSecondBetaIter->isDone() ;
	       theSecondBetaIter->next()) { // loop on beta  parameters
	
	    bioFixedParameter* betaj = theSecondBetaIter->currentItem() ;
	    if (!betaj->isFixedParam()) {
	      if (betaj == NULL) {
		err = new patErrNullPointer("bioFixedParameter") ;
		WARNING(err->describe()) ;
		return ;
	      }
	      patULong j = betaj->getEstimatedParameterId() ;
	    
	      if (betaj->getName() > betai->getName()) {
		patCorrelation c ;
		c.firstCoef = betai->getName() ;
		c.secondCoef = betaj->getName() ;
		patReal vi = betai->getValue() ;
		patReal vj = betaj->getValue() ;
		if (err != NULL) {
		  WARNING(err->describe()) ;
		  return ;
		}
		if (isRaoCramerAvailable()) {
		  patReal varJ = (*theRawResults->varianceCovariance)[j][j] ;
		  c.covariance = (*theRawResults->varianceCovariance)[i][j] ;
		  patReal tmp = varI * varJ ;
		  c.correlation = (tmp > 0) 
		    ? c.covariance / sqrt(tmp) 
		    : 0.0 ;
		  tmp = varI + varJ - 2.0 * c.covariance ;
		  c.ttest = (tmp > 0)
		    ? (vi - vj) / sqrt(tmp) 
		    : 0.0  ;
		}
		if (isSandwichAvailable()) {
		  patReal robustVarJ = (*theRawResults->robustVarianceCovariance)[j][j] ;
		  c.robust_covariance = (*theRawResults->robustVarianceCovariance)[i][j] ;
		  patReal tmp = robustVarI * robustVarJ ;
		  c.robust_correlation = (tmp > 0) 
		    ? c.robust_covariance / sqrt(tmp) 
		    : 0.0 ;
		  tmp = robustVarI + robustVarJ - 2.0 * c.robust_covariance ;
		  c.robust_ttest = (tmp > 0)
		    ? (vi - vj) / sqrt(tmp) 
		    : 0.0  ;
		}
	      
		correlation.insert(c) ;
	      }
	    }
	  }
	}
      }
      htmlFile << "<h2>Correlation of coefficients</h2>" << endl ;
      htmlFile << "<p><font size='-1'>Click on the headers of the columns to sort the table [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>" << endl ;
      htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
      htmlFile << "<tr class=biostyle>" ;
      // Col 1
      htmlFile << "<th>Coefficient1</th>" ;
      // Col 2
      htmlFile << "<th>Coefficient2</th>" ;
      // Col 3
      htmlFile << "<th>Covariance</th>" ;
      // Col 4
      htmlFile << "<th>Correlation</th>" ;
      // Col 5
      htmlFile << "<th>t-test</th>" ;
      // Col 6
      if (printPValue) {
	htmlFile << "<th>p-value</th>" ;
      }
      // Col 7
      htmlFile << "<th></th>" ;
      // Col 8
      htmlFile << "<th>Rob. cov.</th>" ;
      // Col 9
      htmlFile << "<th>Rob. corr.</th>" ;
      // Col 10
      htmlFile << "<th>Rob. t-test</th>" ;
      if (printPValue) {
	htmlFile << "<th>p-value</th>" ;
      }
      // Col 11
      htmlFile << "<th></th>";
      // Col 12
      htmlFile << "</tr>"  << endl ;
  
      for (std::set<patCorrelation,patCompareCorrelation>::iterator i = correlation.begin() ;
	   i != correlation.end() ;
	   ++i) {
	htmlFile << "<tr class=biostyle>" << endl ;
	// Col 1
	htmlFile << "<td>" << i->firstCoef << "</td>" ;
	// Col 2
	htmlFile << "<td>" << i->secondCoef << "</td>" ;
	// Col 3
	htmlFile << "<td>" << theNumber.formatParameters(i->covariance) << "</td>" ;
	// Col 4
	htmlFile << "<td>" << theNumber.formatParameters(i->correlation) << "</td>" ;
	// Col 5
	htmlFile << "<td>" << theNumber.formatTTests(i->ttest) << "</td>" ;
	// Col 6
	if (printPValue) {
	  patReal pvalue = patPValue(patAbs(i->ttest),err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
	}
	// Col 7
	if (patAbs(i->ttest) < tTestThreshold) {
	  htmlFile << "<td>" << warningSign << "</td>" ;
	}
	else {
	  htmlFile << "<td></td>" ;
	}
	if (isSandwichAvailable()) {
	  // Col 8
	  htmlFile << "<td>" << theNumber.formatParameters(i->robust_covariance) << "</td>" ;
	  // Col 9
	  htmlFile << "<td>" << theNumber.formatParameters(i->robust_correlation) << "</td>" ;
	  // Col 10
	  htmlFile << "<td>" << theNumber.formatTTests(i->robust_ttest) << "</td>";
	  // Col 11
	  if (printPValue) {
	  
	    patReal pvalue = patPValue(patAbs(i->robust_ttest),err) ;
	    if (err != NULL) {
	      WARNING(err->describe()) ;
	      return ;
	    }
	    htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>"  ;
	  }
	  // Col 12
	  if (patAbs(i->robust_ttest) < tTestThreshold) {
	    htmlFile << "<td>" << warningSign  << "</td>" ;
	  }
	  else {
	    htmlFile << "<td></td>" ;
	  }
	  
	}
	htmlFile << "</tr>" ;
      }
      htmlFile << "</table>" << endl ;

      htmlFile << "<p>Smallest singular value: " << theRawResults->smallestSingularValue << "</p>" ;
      if (!theRawResults->eigenVectors.empty()) {
	htmlFile << "<h2>Unidentifiable model</h2>" << endl ;
	htmlFile << "<p>The log likelihood is (almost) flat along the following combinations of parameters</p>" << endl ;
	htmlFile << "<table border=\"0\">" << endl ;
    
	patReal threshold = bioParameters::the()->getValueReal("singularValueThreshold",err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	//    patIterator<patBetaLikeParameter>* theParamIter = createAllParametersIterator() ;
    
	for(map<patReal,patVariables>::iterator iter = 
	      theRawResults->eigenVectors.begin() ;
	    iter != theRawResults->eigenVectors.end() ;
	    ++iter) {
	  htmlFile << "<tr class=biostyle><td>Sing. value</td><td>=</td><td>"
		   << iter->first 
		   << "</td></tr>" << endl ; 
      
	  vector<patBoolean> printed(iter->second.size(),patFALSE) ;
	  for (theBetaIter->first() ;
	       !theBetaIter->isDone() ;
	       theBetaIter->next()) { // loop on beta  parameters
	    bioFixedParameter* beta = theBetaIter->currentItem() ;
	    if (beta == NULL) {
	      err = new patErrNullPointer("bioFixedParameter") ;
	      WARNING(err->describe()) ;
	      return ;
	    }
	    if (!beta->isFixedParam()) {
	      patULong j = beta->getEstimatedParameterId() ;
	      
	      if (j >= iter->second.size()) {
		err = new patErrOutOfRange<patULong>(j,0,iter->second.size()-1) ;
		WARNING(err->describe()) ;
		return ;
	      }
	      if (patAbs(iter->second[j]) >= threshold) {
		htmlFile << "<tr class=biostyle><td>" 
			 << iter->second[j] 
			 << "</td><td>*</td><td>" 
			 << beta->getName() << "</td></tr>" << endl ;
		printed[j] = patTRUE ;
	      }
	    }
	  }
	  for (patULong j = 0 ; j < iter->second.size() ; ++j) {
	    if (patAbs(iter->second[j]) >= threshold && !printed[j]) {
	      htmlFile << "<tr class=biostyle><td>" 
		       << iter->second[j] 
		       << "</td><td>*</td><td>Param[" 
		       << j << "]</td></tr>" << endl ;
	  
	    }
	  }
	}


	htmlFile << "</table>" << endl ;
      }
    
    }
  }
  
  if (theBayesianResults != NULL) {
    htmlFile << "<h1>Bayesian estimation report</h1>" << endl ;
    htmlFile << endl ;
    
    
    htmlFile << "<table border=\"0\">" << endl ;
    
    
    patULong nDraws = bioParameters::the()->getValueInt("NbrOfDraws",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    htmlFile << "<tr class=biostyle><td align=right ><strong>Number of draws</strong>:</td> <td>"<< nDraws <<"</td></tr>" << endl ;
    if (nbrParameters != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Number of estimated parameters</strong>: </td> <td>"<< nbrParameters <<"</td></tr>" << endl ;
    }
    if (sampleSize != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Sample size</strong>:</td> <td>"<< sampleSize <<"</td></tr>" << endl ;
    }
    if (removedObservations != patBadId) {
      htmlFile << "<tr class=biostyle><td align=right ><strong>Excluded observations</strong>:</td> <td>"<< removedObservations <<"</td></tr>" << endl ;
    }
    if (runTime != "") {
      htmlFile << "<tr class=biostyle><td align=right><strong>Run time</strong>:	</td> <td>"
	       << runTime
	       <<"</td></tr>" << endl ;
    }

    htmlFile << "</table>" << endl ;
    htmlFile << "<h2>Bayesian estimates of the parameters</h2>" << endl ;
    htmlFile << "<p><font size='-1'>Click on the headers of the columns to sort the table  [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    htmlFile << "<tr class=biostyle>" ;
    // Col 1
    htmlFile << "<th>Name </th>" ;
    // Col 2
    htmlFile << "<th>Value</th>" ;
    // Col 3
    htmlFile << "<th>Std err</th>" ;
    // Col 4
    htmlFile << "<th>t-test</th>" ;
    int printPValue = bioParameters::the()->getValueInt("printPValue",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (printPValue != 0) {
      // Col 5
      htmlFile << "<th>p-value</th>" ;
    }

    patReal tTestThreshold = bioParameters::the()->getValueReal("tTestThreshold",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    patString warningSign = bioParameters::the()->getValueString("warningSign",err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    DEBUG_MESSAGE("nBetas = " << theBayesianResults->nBetas()) ;
    DEBUG_MESSAGE("nDraws = " << theBayesianResults->nDraws()) ;
      
    for (patULong i = 0 ; i < theBayesianResults->nBetas() ; ++i) {
      htmlFile << "<tr class=biostyle>" << endl ;
      htmlFile << "<td>" << theBayesianResults->paramNames[i] << "</td>" ;
      patReal theValue = theBayesianResults->mean[i] ;
      patReal variance = theBayesianResults->varCovar->getElement(i,i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal stdErr = (variance >= 0) ? sqrt(variance) : patMaxReal ;
      patReal ttest ;
      if (stdErr != patMaxReal) {
	ttest = theValue / stdErr ; 
      }
      else {
	ttest = 0.0 ;
      }
      htmlFile << "<td>" << theNumber.formatParameters(theValue) << "</td>" ;
      htmlFile << "<td>" << theNumber.formatParameters(stdErr) << "</td>";
      htmlFile << "<td>" << theNumber.formatTTests(ttest) << "</td>" ;
      if (printPValue) {
	patReal pvalue = patPValue(patAbs(ttest),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
      }
      if (patAbs(ttest) < tTestThreshold || !patFinite(ttest)) {
	htmlFile << "<td>" << warningSign << "</td>";
      }
      else {
	htmlFile << "<td></td>" ;
      }
      htmlFile << "</tr>" << endl;
    }
    htmlFile << "</table>" ;

    std::set<patCorrelation,patCompareCorrelation> correlation ;
    for (patULong i = 0 ; i < theBayesianResults->nBetas() ; ++i) {
      patReal varianceI = theBayesianResults->varCovar->getElement(i,i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      for (patULong j = i+1 ; j < theBayesianResults->nBetas() ; ++j) {
	patReal varianceJ = theBayesianResults->varCovar->getElement(j,j,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	patCorrelation c ;
	c.firstCoef = theBayesianResults->paramNames[i] ;
	c.secondCoef = theBayesianResults->paramNames[j] ;
	patReal vi = theBayesianResults->mean[i] ;
	patReal vj = theBayesianResults->mean[j] ;
	c.covariance = theBayesianResults->varCovar->getElement(i,j,err) ;
	patReal tmp = varianceI * varianceJ ;
	c.correlation = (tmp > 0) 
	  ? c.covariance / sqrt(tmp) 
	  : 0.0 ;
	tmp = varianceI + varianceJ - 2.0 * c.covariance ;
	c.ttest = (tmp > 0)
	  ? (vi - vj) / sqrt(tmp) 
	  : 0.0  ;
	
	correlation.insert(c) ;
	
      }
    }

    htmlFile << "<h2>Correlation of coefficients</h2>" << endl ;
    htmlFile << "<p><font size='-1'>Click on the headers of the columns to sort the table [<a href='http://www.kryogenix.org/code/browser/sorttable/' target='_blank'>Credits</a>]</font></p>" << endl ;
    htmlFile << "<table border=\"1\" class=\"sortable\">" << endl ;
    htmlFile << "<tr class=biostyle>" ;
    // Col 1
    htmlFile << "<th>Coefficient1</th>" ;
    // Col 2
    htmlFile << "<th>Coefficient2</th>" ;
    // Col 3
    htmlFile << "<th>Covariance</th>" ;
    // Col 4
    htmlFile << "<th>Correlation</th>" ;
    // Col 5
    htmlFile << "<th>t-test</th>" ;
    // Col 6
    if (printPValue) {
      htmlFile << "<th>p-value</th>" ;
    }
    // Col 7
    htmlFile << "<th></th>" ;
    
    htmlFile << "</tr>"  << endl ;
  
    for (std::set<patCorrelation,patCompareCorrelation>::iterator i = correlation.begin() ;
	 i != correlation.end() ;
	 ++i) {
      htmlFile << "<tr class=biostyle>" << endl ;
      // Col 1
      htmlFile << "<td>" << i->firstCoef << "</td>" ;
      // Col 2
      htmlFile << "<td>" << i->secondCoef << "</td>" ;
      // Col 3
      htmlFile << "<td>" << theNumber.formatParameters(i->covariance) << "</td>" ;
      // Col 4
      htmlFile << "<td>" << theNumber.formatParameters(i->correlation) << "</td>" ;
      // Col 5
      htmlFile << "<td>" << theNumber.formatTTests(i->ttest) << "</td>" ;
      // Col 6
      if (printPValue) {
	patReal pvalue = patPValue(patAbs(i->ttest),err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" << theNumber.formatTTests(pvalue) << "</td>" ;
      }
      // Col 7
      if (patAbs(i->ttest) < tTestThreshold) {
	htmlFile << "<td>" << warningSign << "</td>" ;
      }
      else {
	htmlFile << "<td></td>" ;
      }
      htmlFile << "</tr>" ;
    }
    htmlFile << "</table>" << endl ;
    
  }
  
  if (simulatedValues != NULL) {
    patULong oldPrecision = htmlFile.precision() ;
    patULong floatPointPrecision = bioParameters::the()->getValueInt("decimalPrecisionForSimulation",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    htmlFile << setprecision(floatPointPrecision) ;
    htmlFile << "<h1>Simulation report</h1>" << endl ;
    htmlFile << endl ;
    htmlFile << "<p>Number of draws for Monte-Carlo: " <<  bioParameters::the()->getValueInt("NbrOfDraws",err) << "</p>" << endl ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    htmlFile << "<p>Type of draws: " <<   bioParameters::the()->getValueString("RandomDistribution",err) << "</p>" << endl ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (sensitivityAnalysis) {
      htmlFile << "<p>Number of draws for sensitivity analysis: " << bioParameters::the()->getValueInt("NbrOfDrawsForSensitivityAnalysis",err) << "</p>" << endl ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    
    
    htmlFile << "<h2>Aggregate values</h2>" ;
    htmlFile << "<table border='1'>" << endl ;
    htmlFile << "<tr><th></th>" ;

    patString weightHeader = bioParameters::the()->getValueString("HeaderForWeightInSimulatedResults",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return  ;
    }
    
    map<patString,bioSimulatedValues>::iterator found =
      simulatedValues->find(weightHeader) ;
    patBoolean withWeight(found != simulatedValues->end()) ;
    vector<patReal> theWeights ;
    if (withWeight) {
      for (patULong i = 0 ; i < found->second.getNumberOfValues() ; ++i) {
	theWeights.push_back(found->second.getNominalValue(i)) ;
      }
    }
    
    for (map<patString,bioSimulatedValues>::iterator i =
	   simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      if (withWeight) {
	i->second.setWeights(&theWeights) ;
      }
      i->second.calculateConfidenceIntervals(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }    

    patReal alpha = bioParameters::the()->getValueReal("sensitivityAnalysisAlpha",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    

    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<th>" << i->first << "</th>" ;
      if (i->second.hasSimulatedValues()) {
	int a = int(100.0 * alpha) ;
	htmlFile << "<th colspan='2'>" << i->first << ": " << 100-2*a << "% CI</th>" ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
      }
    }
    htmlFile << "</tr>" ;

    htmlFile << "<tr>" ;
    htmlFile << "<th>Total</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getTotal(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	patInterval ci = i->second.getTotalConfidenceInterval(err)  ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" 
		 << ci.getLowerBound() 
		 << "</td><td>" 
		 << ci.getUpperBound() 
		 << "</td>" ;
	
      }
    }
    
    htmlFile << "</tr>" ;

    if (withWeight) {
      htmlFile << "<tr>" ;
      htmlFile << "<th>Weighted total</th>" ;
      for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	   i != simulatedValues->end() ;
	   ++i) {
	htmlFile << "<td>" << i->second.getWeightedTotal(err) << "</td>" ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	if (i->second.hasSimulatedValues()) {
	  patInterval ci = i->second.getWeightedTotalConfidenceInterval(err)  ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" 
		   << ci.getLowerBound() 
		   << "</td><td>" 
		   << ci.getUpperBound() 
		   << "</td>" ;
	  
	}
      }
    
      htmlFile << "</tr>" ;
    }

    htmlFile << "<tr>" ;
    htmlFile << "<th>Average</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getAverage(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	patInterval ci = i->second.getAverageConfidenceInterval(err)  ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" 
		 << ci.getLowerBound() 
		 << "</td><td>" 
		 << ci.getUpperBound() 
		 << "</td>" ;
	
      }
    }

    htmlFile << "</tr>" ;

    if (withWeight) {
      htmlFile << "<tr>" ;
      htmlFile << "<th>Weighted average</th>" ;
      for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	   i != simulatedValues->end() ;
	   ++i) {
	htmlFile << "<td>" << i->second.getWeightedAverage(err) << "</td>" ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	if (i->second.hasSimulatedValues()) {
	  patInterval ci = i->second.getWeightedAverageConfidenceInterval(err)  ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" 
		   << ci.getLowerBound() 
		   << "</td><td>" 
		   << ci.getUpperBound() 
		   << "</td>" ;
	  
	}
      }
    
      htmlFile << "</tr>" ;
    }


    htmlFile << "<tr>" ;
    htmlFile << "<th>Non zeros</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getNonZeros(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	htmlFile << "<td></td><td></td>" ;
      }
    }

    htmlFile << "</tr>" ;

    htmlFile << "<tr>" ;
    htmlFile << "<th>Non zeros average</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getNonZeroAverage(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	patInterval ci = i->second.getNonZeroAverageConfidenceInterval(err)  ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" 
		 << ci.getLowerBound() 
		 << "</td><td>" 
		 << ci.getUpperBound() 
		 << "</td>" ;
	
      }
    }

    htmlFile << "</tr>" ;

    if (withWeight) {
      htmlFile << "<tr>" ;
      htmlFile << "<th>Weighted non zeros average</th>" ;
      for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	   i != simulatedValues->end() ;
	   ++i) {
	htmlFile << "<td>" << i->second.getWeightedNonZeroAverage(err) << "</td>" ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	if (i->second.hasSimulatedValues()) {
	  patInterval ci = i->second.getWeightedNonZeroAverageConfidenceInterval(err)  ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return ;
	  }
	  htmlFile << "<td>" 
		   << ci.getLowerBound() 
		   << "</td><td>" 
		   << ci.getUpperBound() 
		   << "</td>" ;
	  
	}
      }
    
      htmlFile << "</tr>" ;
    }

    htmlFile << "<tr>" ;
    htmlFile << "<th>Minimum</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getMinimum(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	patInterval ci = i->second.getMinimumConfidenceInterval(err)  ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" 
		 << ci.getLowerBound() 
		 << "</td><td>" 
		 << ci.getUpperBound() 
		 << "</td>" ;
	
      }
    }

    htmlFile << "</tr>" ;


        htmlFile << "<tr>" ;
    htmlFile << "<th>Maximum</th>" ;
    for (map<patString,bioSimulatedValues >::iterator i = simulatedValues->begin() ;
	 i != simulatedValues->end() ;
	 ++i) {
      htmlFile << "<td>" << i->second.getMaximum(err) << "</td>" ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (i->second.hasSimulatedValues()) {
	patInterval ci = i->second.getMaximumConfidenceInterval(err)  ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return ;
	}
	htmlFile << "<td>" 
		 << ci.getLowerBound() 
		 << "</td><td>" 
		 << ci.getUpperBound() 
		 << "</td>" ;
	
      }
    }

    htmlFile << "</tr>" ;


    
    htmlFile << "</table>" ;

    
    patBoolean reportAllRecords = bioParameters::the()->getValueInt("simulateReportForEveryObservation",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (reportAllRecords) {
      htmlFile << "<h2>Detailed records</h2>" ;
      htmlFile << "<table border='1'>" << endl ;
      patULong size ;
      // It is assumed that the size of each vector is the same
    
    

      htmlFile << "<tr>" ;
      htmlFile << "<th>Row</th>" ;
      for (map<patString,bioSimulatedValues>::iterator i = simulatedValues->begin() ;
	   i != simulatedValues->end() ;
	   ++i) {
	size = i->second.getNumberOfValues() ;
	if (reportAllRecords) {
	  htmlFile << "<th>" << i->first << "</th>" ;
	  if (i->second.hasSimulatedValues()) {
	    int a = int(100.0 * alpha) ;
	    htmlFile << "<th colspan='2'>" << i->first << ": " << 100-2*a << "% CI</th>" ;
	  }
	}
      }
      htmlFile << "</tr>" ;
      for (patULong i = 0 ; i < size ; ++i) {
	htmlFile << "<tr>" ;
	htmlFile << "<td>" << i+1 << "</td>" ;
	for (map<patString,bioSimulatedValues>::iterator j = 
	       simulatedValues->begin() ;
	     j != simulatedValues->end() ;
	     ++j) {
	  if (i < j->second.getNumberOfValues()) {
	    if (reportAllRecords) {
	      htmlFile << "<td>" << j->second.getNominalValue(i) << "</td>" ;
	      if (j->second.hasSimulatedValues()) {
		htmlFile << "<td>" 
			 << j->second.getConfidenceInterval(i).getLowerBound() 
			 << "</td><td>" 
			 << j->second.getConfidenceInterval(i).getUpperBound() 
			 << "</td>" ;
	      }
	    }
	  }
	  else {
	    htmlFile << "<td>---</td>" ;
	  }
	}
	htmlFile << "</tr>" ;
      }
      htmlFile << "</table>" << endl ;

    }


    if (mcNoCorrection.size() != 0) {
      htmlFile << "<h1>Precision of Monte-Carlo simulation for integrals</h1>" << endl ;
      htmlFile << "<dl>" << endl ;
      htmlFile << "<dt>Value without correction</dt>" << endl ;
      htmlFile << "<dd>Output of the Monte-Carlo simulation (vmc).</dd>" << endl ;
      htmlFile << "<dt>Value with control variate correction</dt>" << endl ;
      htmlFile << "<dd>Output of the Monte-Carlo simulation after the application of the control variate method (vcv). Value used by Biogeme.</dd>" << endl ;
      htmlFile << "<dt>Relative error</dt>" << endl ;
      htmlFile << "<dd>100 (vmc - vcv) / vcv</dd>" << endl ;
      htmlFile << "<dt>Std. dev. without correction</dt>" << endl ;
      htmlFile << "<dd>Calculated as the square root of the variance of the original draws, divided by the square root of the number of draws (stdmc).</dd>" << endl ;
      htmlFile << "<dt>Std. dev. with control variate correction</dt>" << endl ;
      htmlFile << "<dd>Calculated as the square root of the variance of the corrected draws, divided by the square root of the number of draws (stdcv).</dd>" << endl ;
      htmlFile << "<dt>Reduced number of draws</dt>" << endl ;
      htmlFile << "<dd>Rcv = R stdcv<sup>2</sup>/stdmv<sup>2</sup>, where R is the current number of draws. This is the number of draws that are sufficient (when the correction is applied) to achieve the same precision as the method without correction.</dd>" << endl ;
      htmlFile << "<dt>Savings</dt>" << endl ;
      htmlFile << "<dd>100 * (R-Rcv) / R</dd>" << endl ;
      htmlFile << "<dt>Control variate simulated</dt>" << endl ;
      htmlFile << "<dd>Value of the integral used for control variate using Monte-Carlo simulation (simulated). </dd>" << endl ;
      htmlFile << "<dt>Control variate analytical</dt>" << endl ;
      htmlFile << "<dd>Value of the analytical integral used for control variate (analytical).</dd>" << endl ;
      htmlFile << "<dt>Relative error on control variate</dt>" << endl ;
      htmlFile << "<dd>100 (simulated - analytical) / analytical. If this value is more than 1% (in absolute value), the row is displayed in red, emphasizing that either the number of draws for the original Monte-Carlo is insufficient, or the analytical value of the integral is wrong. </dd>" << endl ;
      htmlFile << "</dl>" << endl ;
      htmlFile << "<p><strong>Number of draws: " << nbrDraws << "</strong></p>" << endl ;
      htmlFile << "<table border='1'>" << endl ;
      htmlFile << "<tr>" ;
      htmlFile << "<th>Value without correction</th>" ;
      htmlFile << "<th>Value with control variate correction</th>" ;
      htmlFile << "<th>Relative error</th>" ;
      htmlFile << "<th>Std. dev. without correction</th>" ;
      htmlFile << "<th>Std. dev. with control variate correction</th>" ;
      htmlFile << "<th>Reduced number of draws</th>" ;
      htmlFile << "<th>Savings</th>" ;
      htmlFile << "<th>Control variate simulated</th>" ;
      htmlFile << "<th>Control variate analytical</th>" ;
      htmlFile << "<th>Relative error on control variate</th>" ;
      htmlFile << "</tr>" ;
      for (patULong i = 0 ; i < mcNoCorrection.size()  ; ++i) {
	patReal relError = 100.0 * (mcSimulated[i]-mcAnalytical[i])/mcAnalytical[i] ;
	if (patAbs(relError) > 1.0) {
	    htmlFile << "<tr class=boundstyle>" << endl ;
	}
	else {
	    htmlFile << "<tr class=biostyle>" << endl ;
	}
	htmlFile << "<td>" ;
	htmlFile << mcNoCorrection[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << mcCorrected[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	if (mcCorrected[i] != 0.0) {
	  htmlFile << 100.0 *(mcNoCorrection[i] - mcCorrected[i]) / mcCorrected[i] << "%" ;
	}
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << mcStdDev[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << mcCorrectedStdDev[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	patReal cvDraws = 
	  nbrDraws * mcCorrectedStdDev[i] * mcCorrectedStdDev[i] / (mcStdDev[i] * mcStdDev[i]) ;
	if (mcStdDev[i] != 0.0) {
	  htmlFile << patULong(cvDraws);
	}
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	if (mcStdDev[i] != 0.0) {
	  htmlFile << 100.0 * patReal(nbrDraws-cvDraws) / patReal(nbrDraws) << "%" ;
	}
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << mcSimulated[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << mcAnalytical[i] ;
	htmlFile << "</td>" ;
	htmlFile << "<td>" ;
	htmlFile << relError << "%" ;
	htmlFile << "</td>" ;
	htmlFile << "</tr>" ;
      }
      htmlFile << "</table>" << endl ;
    }
    
    htmlFile << "</body>" << endl;
    htmlFile << "</html>" << endl;
    htmlFile << setprecision(oldPrecision) ;
  }

  htmlFile.close() ;
  patOutputFiles::the()->addCriticalFile(fileName,"Results in HTML format");
		  
  
}


void bioReporting::computeFromSample(bioSample* s, patError*& err) {

  if (s == NULL) {
    err = new patErrNullPointer("bioSample") ;
    WARNING(err->describe()) ;
    return ;
  }
  sampleFile = s->getDataFileName() ;
  sampleSize = s->size() ;
  removedObservations = s->getNbrExcludedRow() ;
  dataProcessingTime = s->getProcessingTime() ;
}

void bioReporting::setDiagnostic(patString diag) {
  diagnostic = diag ;
}

void bioReporting::setIterations(patULong i) {
  iterations = i ;
}

void bioReporting::setRunTime(patString t) {
  runTime = t ;
}

void bioReporting::setDataProcessingTime(patString t) {
  dataProcessingTime = t ;
}

void bioReporting::setInitLL(patReal l) {
  initLoglikelihood = l ;
}

void bioReporting::addStatistic(patString name, patReal value) {
  statistics[name] = value ;
}

void bioReporting::addFormula(patString name, patString formula) {
  formulas[name] = formula ;
}

void bioReporting::setSimulationResults(map<patString,bioSimulatedValues>* s, 
					patBoolean sensitivity){
  simulatedValues = s ;
  sensitivityAnalysis = sensitivity ;
} 


void bioReporting::printEstimatedParameters(patString filename, patError*& err) {

  ofstream pyEst(filename.c_str()) ;
  pyEst << "# This file has automatically been generated" << endl ; 
  pyEst << "# " << bioVersion::the()->getVersionInfoDate() << endl ;
  pyEst << "# " << bioVersion::the()->getVersionInfoAuthor() << endl ;
  patAbsTime now ;
  now.setTimeOfDay() ;
  pyEst << "# " << now.getTimeString(patTsfFULL) << "</p>" << endl ;
  pyEst << "#" << endl ;
  pyEst << bioLiteralRepository::the()->printFixedParameters(patTRUE) << endl ;

  pyEst << "#" ;
  pyEst << "# Code for the sensitivity analysis" << endl;
  pyEst << "names = [" ;
  patIterator<bioFixedParameter*>* theBetaIter = bioLiteralRepository::the()->getSortedIteratorFixedParameters() ;
  patIterator<bioFixedParameter*>* theSecondBetaIter = bioLiteralRepository::the()->getSortedIteratorFixedParameters() ;
  patBoolean firstBeta = patTRUE ;
  for (theBetaIter->first() ;
       !theBetaIter->isDone() ;
       theBetaIter->next()) { // loop on beta  parameters
    
    bioFixedParameter* beta = theBetaIter->currentItem() ;
    if (!beta->isFixedParam()) {
      if (firstBeta) {
	firstBeta = patFALSE ;
      }
      else {
	pyEst << "," ;
      }
      pyEst << "'" << beta->getName() << "'" ;
    }
  }  
  pyEst << "]" << endl ;
  pyEst << "values = [" ;
  patBoolean theFirsti = patTRUE ;
  for (theBetaIter->first() ;
       !theBetaIter->isDone() ;
       theBetaIter->next()) { // loop on beta  parameters
    bioFixedParameter* betai = theBetaIter->currentItem() ;
    if (!betai->isFixedParam()) {
      patULong i = betai->getEstimatedParameterId() ;
      if (betai == NULL) {
	err = new patErrNullPointer("bioFixedParameter") ;
	WARNING(err->describe()) ;
	return ;
      }
      if (theFirsti) {
	theFirsti = patFALSE ;
      }
      else {
	pyEst << "," ;
      }
      pyEst << "[" ;
      patBoolean theFirstj = patTRUE ;
      for (theSecondBetaIter->first() ;
	   !theSecondBetaIter->isDone() ;
	   theSecondBetaIter->next()) { // loop on beta  parameters
	
	bioFixedParameter* betaj = theSecondBetaIter->currentItem() ;
	if (!betaj->isFixedParam()) {
	  if (betaj == NULL) {
	    err = new patErrNullPointer("bioFixedParameter") ;
	    WARNING(err->describe()) ;
	    return ;
	  }
	  patULong j = betaj->getEstimatedParameterId() ;
	  if (theFirstj) {
	    theFirstj = patFALSE ;
	  }
	  else {
	    pyEst << "," ;
	  }
	  pyEst << (*theRawResults->robustVarianceCovariance)[i][j] ; 
	}
      }
      pyEst << "]" ;
    }
  }
  DELETE_PTR(theBetaIter) ;
  DELETE_PTR(theSecondBetaIter) ;
  pyEst << "]" << endl ;
  pyEst << "vc = bioMatrix(" << bioLiteralRepository::the()->getNumberOfEstimatedParameters() << ",names,values)" << endl ;
  pyEst << "BIOGEME_OBJECT.VARCOVAR = vc" << endl ;
  
  pyEst.close() ;
  patOutputFiles::the()->addCriticalFile(filename,"Results in Python format (to copy/paste for simulation)");
  

  GENERAL_MESSAGE("Estimated parameters: " << endl << bioLiteralRepository::the()->printFixedParameters()) ;
  GENERAL_MESSAGE("File " << filename << " created") ;


}

void bioReporting::computeBayesianResults(bioBayesianResults* rr,patError*& err) {

  if (rr == NULL) {
    err = new patErrNullPointer("bioBayesianResults") ;
    WARNING(err->describe()) ;
    return ;
  }
  theBayesianResults = rr ;
  theBayesianResults->computeStatistics(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
}

void bioReporting::addMonteCarloReport(patStatistics* mainDraws,
				       patStatistics* cvDraws,
				       patKalman* theFilter,
				       patReal analytical, patError*& err) {

  if (!parameterRead) {
    mcReporting = bioParameters::the()->getValueInt("monteCarloControlVariateReport",err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    parameterRead = patTRUE ;
  }

  if (mainDraws == NULL) {
    err = new patErrNullPointer("patStatistics") ;
    WARNING(err->describe()) ;
    return ;
  }

  nbrDraws = mainDraws->getSize() ;
  if (cvDraws == NULL) {
    err = new patErrNullPointer("patStatistics") ;
    WARNING(err->describe()) ;
    return ;
  }
  if (theFilter == NULL) {
    err = new patErrNullPointer("patKalman") ;
    WARNING(err->describe()) ;
    return ;
  }
  

  if (mcNoCorrection.size() < mcReporting) {

    mcNoCorrection.push_back(mainDraws->getMean()) ;
    mcCorrected.push_back(theFilter->evaluate(analytical)) ;
    mcStdDev.push_back(sqrt(mainDraws->getVariance())/sqrt(patReal(mainDraws->getSize()))) ;
    patReal a = theFilter->getCoefficient() ;
    patReal newVar = mainDraws->getVariance() - a * a * cvDraws->getVariance() ;
      mcCorrectedStdDev.push_back(sqrt(patMax(newVar,0.0))/sqrt(patReal(mainDraws->getSize()))) ;
    mcSimulated.push_back(cvDraws->getMean()) ;
    mcAnalytical.push_back(analytical) ;
      
  }
}

void bioReporting::involvesMonteCarlo() {
  reportDraws = patTRUE ;
}


void bioReporting::writeALogit(patString fileName, patError*& err) {

  err = new patErrMiscError("To be implemented") ;
  ofstream alogitFile(fileName.c_str()) ;

  alogitFile.close() ;
  patOutputFiles::the()->addUsefulFile(fileName,"Results in ALOGIT format");
  GENERAL_MESSAGE("File " << fileName << " created") ;
}

 void bioReporting::setAlgorithm(patString a) {
   algorithm = a ; 
 }
