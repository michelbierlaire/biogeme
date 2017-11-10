//-*-c++-*------------------------------------------------------------
//
// File name : bioParameters.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Jul 17 14:17:08 2009
//
//--------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "patDisplay.h"
#include "patOutputFiles.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "bioParameters.h"
#include "bioPythonSingletonFactory.h"

bioParameters* bioParameters::the() {
  return bioPythonSingletonFactory::the()->bioParameters_the() ;
}

bioParameters::bioParameters() : docFile(patString("pythonparam.html")) {
  stringValues["OutputFileForSensitivityAnalysis"] =
    pair<patString,patString>(patString("__sensitivity.dat"),
			      patString("Biogeme performs sensitivity analysis by simulation. If asked by the user, each simulated value is stored in the file with the name defined by this parameter.")) ;

  stringValues["HeaderForWeightInSimulatedResults"] = 
    pair<patString,patString>(patString("__weight"),
			      patString("Biogeme includes the weight of each observation in the simulation results. This parameters sets the label of the resulting column in the output.")) ;

  stringValues["HeaderOfRowId"] = 
    pair<patString,patString>(patString("__rowId__"),
			      patString("Biogeme assumes that the data file contains a virtual column where the number of the row is reported. The name of this column is defined by this parameter, and can be used in the specification.")) ;
			      
  stringValues["RandomDistribution"] = 
    pair<patString,patString>(patString("MLHS"),
			      patString("Type of draws used for simulating random distributions. Valid values are <dl><dt><code>PSEUDO</code></dt><dd>Pseudo random numbers</dd><dt><code>HALTON</code></dt><dd>Halton sequences</dd><dt><code>MLHS</code></dt><dd> Modified Latin Hypercube Sampling (see <a href='http://dx.doi.org/10.1016/j.trb.2004.10.005'  target='_blank'>Hess et al., 2006</a>).</dd></dl>")) ;

  stringValues["warningSign"] = 
    pair<patString,patString>(patString("*"),
			      patString("Signs printed in the estimation report in front of parameters such that the <em>t</em>-test is between -1.96 and 1.96 (or the value defined by the parameter <code>tTestThreshold</code>).")) ;

  stringValues["optimizationAlgorithm"] = 
    pair<patString,patString>(patString("BIO"),
			      patString("Name of the optimization algorithm used to solve the maximum likelihood estimation problem. Valid values are:<dl><dt>BIO</dt><dd>Trust region method implemented for Biogeme</dd><dt>CFSQP</dt><dd>SQP method by <a href='http://hdl.handle.net/1903/5496' target='_blank'>Craig et al. (1994)</a></dd><dt>IPOPT</dt><dd><a href='https://projects.coin-or.org/Ipopt' target='_blank'>COIN-OR IPOPT</a></dd><dt>SOLVOPT</dt><dd>Implementation of Shor's method by <a href='http://www.kfunigraz.ac.at/imawww/kuntsevich/solvopt/ps/manual.pdf' target='_blank'>Kuntsevich and Kappel (1997)</a></dd></dl>Note that BIOMC is not yet available in Python Biogeme.")) ;

  stringValues["individualId"] =
    pair<patString, patString>(patString("__rowId__"),patString("The name of the variable that identifies the individuals in the sample. It is used by the Bayesian estimation procedure, that generates realizations of random parameters for each individual (see Train, 2003, chap. 12)")) ;



  stringValues["stopFileName"] = 
    pair<patString,patString>(patString("STOP"),
			      patString("During the optimization process, Biogeme checks for the existence of a file, whose name is defined by this parameter. If the file exists, Biogeme interrupts the iterations and generate output files. This is convenient to prematurely stop iterations without loosing the computations performed thus far.")) ;

 integerValues["includeDataWhenSimulate"] =
    pair<long int,patString>(0,
			    patString("When pythonbiogeme is used to simulate, each variable in the data file are included in the report, if the parameter is set to 1."));

 integerValues["accessFirstDataFromMetaIterator"] =
    pair<long int,patString>(0,
			    patString("If the parameter is set to a value different from 0, the first row of a meta iterator can be accessed outside of a rwo iterator. This should be used with caution, as it is in general not the desired effect, and it may hide implementation errors."));



  integerValues["deriveAnalyticalHessian"] =
    pair<long int,patString>(1,
			    patString("The second derivative matrix of the likelihood function is derived if this parameters is different from 0."));

  integerValues["useAnalyticalHessianForOptimization"] =
    pair<long int,patString>(0,
			    patString("If the second derivative matrix is available, it is used in the optimization algorithm if the parameter is different from 0."));


  integerValues["DumpSensitivityAnalysis"] =
    pair<long int,patString>(0,
			    patString("If this parameter is non zero, Biogeme dumps the values simulated during sensitivity analysis in a file."));

  integerValues["NbrOfDraws"] = 
    pair<long int,patString>(1000,
			    patString("Number of draws for the simulation of the random parameters"));

  integerValues["NbrOfDrawsForSensitivityAnalysis"] = 
    pair<long int,patString>(100,
			    patString("Number of draws for the empirical sensitivity analysis"));

  integerValues["Seed"] = 
    pair<long int,patString>(9021967,
			     patString("Seed for random number generation.")) ;

  integerValues["printPValue"] = 
    pair<long int,patString>(1,
			     patString("If 1, Biogeme prints the <em>p</em>-value in the results. The <em>p</em>-value is computed as follows: if <em>t</em> is the <em>t</em>-test of the parameters, <em>p = 2 (1 - &Phi;(t))</em>,where <em>&Phi;()</em> is the cumulative density function of the univariate normal distribution.")) ;

  integerValues["monteCarloControlVariateReport"] = 
    pair<long int,patString>(100,
			     patString("Include reporting about the added value of the control variate method in terms of precision for k Monte-Carlo calculations, where k is the value of the parameter")) ;

  integerValues["decimalPrecisionForSimulation"] = 
    pair<long int,patString>(6,
			     patString("Sets the decimal precision to be used to format floating-point values on output operations with the SIMULATE function.")) ;

  integerValues["allowNestedMonteCarlo"] =
    pair<long int,patString>(0,
			     patString("If different from 0, nested MonteCarlo statements are allowed in a formula. In general, the nesting is due to a syntax error, and not explicitly desired. Therefore, the default value is 0.")) ;

  integerValues["computeInitLoglikelihood"] = 
    pair<long int,patString>(1,
			     patString("If 1, Biogeme computes the log likelihood at the starting point, before running the algorithm.")) ;

  integerValues["checkDerivatives"] = 
    pair<long int, patString>(0,
			      patString("If set to 1, the analytical derivatives of the log likelihood functions and the nonlinear constraints are compared to the finite difference derivatives. This is used mainly for debugging purposes."));

#ifdef DEBUG
  integerValues["debugDerivatives"] = 
    pair<long int, patString>(0,
			      patString("If set to 1, the analytical derivatives of each expression are analyzed and the program is the interrupted. Use for debug purposes. Most users do not need this parameter."));
#endif

  realValues["toleranceCheckDerivatives"] = 
    pair<patReal, patString>(0.001,
			      patString("Difference between analytical and finite differences derivative is considered to be significant if greater or equal to this value"));

  integerValues["buildAnalyticalGradient"] = 
    pair<long int,patString>(1,
			     patString("If 1, Biogeme generates the analytical gradient of the log likelihood. If 0, the finite difference approximation is used instead."));

  integerValues["varCovarFromBHHH"] = 
    pair<long int,patString>(0,
			     patString("The computation of the variance-covariance matrix of the estimated parameters using finite difference approximation may take a while for complex models. It is sometimes useful to use the BHHH approximation, which is much faster to compute. If so, set this parameter to 1. It is recommended not to use BHHH in the final model. ")) ;

  integerValues["numberOfThreads"] = 
    pair<long int,patString>(0,
			     patString("This parameter specifies the number of <a href='http://en.wikipedia.org/wiki/Thread_(computer_science)' target='_blank'>threads</a> that will be launched. Note that it may exceed the actual number of available processors. However, this may affect the performance by creating unnecessary overhead. If the value is 0 (which is the case by default), the number of threads is set to the number of processors, multiplied by the value of the parameter shareOfProcessors / 100. If this number is unavailable, the number of threads is set to 4.")) ;

integerValues["shareOfProcessors"] =
  pair<long int,patString>(50,patString("When the parameter numberOfThreads is set to 0, the number of threads is set to the number of available threads reported by the system, multiplied by the value of this parameter, divided by 100. Zero and negative values are ignored. Because of hyperthreading, some systems report more threads than processors physically available")) ;

  integerValues["dumpDrawsOnFile"] =
    pair<long int,patString>(0,patString("If set to 1, Biogeme dumps the draws used for simulated likelihood estimation in the file <code>draws.lis</code>. Another file called <code>model.udraws</code>, where <code>model</code> is the name of the model estimated, contains the uniform draws. ")) ;

  integerValues["saveUniformDraws"] =
    pair<long int,patString>(0,patString("If 1, Biogeme saves the uniform draws used to generate the actual draws. It is used for the test proposed by <a href=\"http://dx.doi.org/10.1016/j.trb.2007.01.002\">Fosgerau and Bierlaire (2007)</a>. It is automatically set to 1 if the operator bioRecycleDraws is used in the model specification.")) ;

  integerValues["warnsForIllegalElements"] =
    pair<long int,patString>(1,patString("The expression <code>Elem</code> is designed to access an element from a dictionary. If Biogeme tries to access an element not in the dictionary, it will trigger a warning if the parameter is set to 1, and stay silent if set to 0.")) ;

  integerValues["cfsqpMode"] =
    pair<long int,patString>(100,patString("Even if it is a descent algorithm, CFSQP sometimes allows non-monotone iterates, hoping not to be trapped in local minima. If the function is convex, a descent algorithm is more appropriate. In this case, set the value to 100. See <a href='http://hdl.handle.net/1903/5496' target='_blank'>CFSQP manual</a> for more details.")) ;

  integerValues["cfsqpIprint"] =
    pair<long int,patString>(2,patString(" Set it to  1 for silent mode, and to 2 for information at each iteration of the optimization algorithm.")) ;

  integerValues["cfsqpMaxIter"] =
    pair<long int,patString>(1000,patString("Maximum number of iterations for CFSQP.")) ;

  integerValues["svdMaxIter"] =
    pair<long int,patString>(150,patString("Maximum number of iterations for the singular value decomposition of the final second derivative matrix.")) ;

  integerValues["moreRobustToNumericalIssues"] =
    pair<long int,patString>(0,patString("Biogeme performs additional verifications about numerical issues (division by zero, log of zero, overflows, etc.) if the parameter is set to 1. It is at the cost of computing performance. ")) ;

  integerValues["scaleDerivativesInSums"] =
    pair<long int,patString>(0,patString("When calculating derivatives (first and second) of sum, Biogeme divide each element by the value of the function before accumulating them, and re-multiply the final result by the same value. It avoids the accumulation of large numbers, that may go beyond the maximum value that can be represented in memory. Note that it is at the expense of performance.")) ;


  integerValues["maxPrimeNumbers"] =
    pair<long int,patString>(1000,patString("The generation of Halton sequences is based on prime numbers. This parameter defines the maximum number of prime numbers that can be used. Most users will never have to change the default value.")) ;

  integerValues["firstIdOfLiterals"] =
    pair<long int,patString>(10000,patString("Each literal is assigned a unique ID by Biogeme. The first of these IDs is defined by this parameter. It is recommended not to modify it. ")) ;


  integerValues["BTRMaxIter"] =
    pair<long int,patString>(1000,patString("Maximum number of iterations to be  performed by the BIO algorithm.")) ;

  integerValues["BTRInitQuasiNewtonWithTrueHessian"] =
    pair<long int,patString>(1,patString("If 1, the secant update is initialized with the analytical hessian (BIO algorithm).")) ;

  integerValues["BTRInitQuasiNewtonWithBHHH"] =
    pair<long int,patString>(1,patString("If 1, the secant update is initialized with the BHHH  (see <a href='http://www.nber.org/chapters/c10206.pdf'>Berndt et al, 1974</a>) approximation (BIO algorithm).")) ;

  integerValues["BTRSignificantDigits"] =
    pair<long int,patString>(7,patString("")) ;

  integerValues["BTRExactHessian"] =
    pair<long int,patString>(1,patString("")) ;

  integerValues["BTRCheapHessian"] =
    pair<long int,patString>(1,patString("If 1,  BHHH (see <a href='http://www.nber.org/chapters/c10206.pdf'>Berndt et al, 1974</a>) is used as an approximation of the second derivatives matrix.")) ;

  integerValues["BTRStartDraws"] =
    pair<long int,patString>(10,patString("If BIOMC is used for  simulated maximum likelihood estimation, this  parameter defines the number of draws which are used during the first iterations. (Not yet implemented)")) ;

  integerValues["BTRIncreaseDraws"] =
    pair<long int,patString>(2,patString(" If BIOMC is used for  simulated maximum likelihood estimation, this  parameters defines the factor by which  the number of draws  is increased. (Not yet implemented)")) ;

  integerValues["maxGcpIter"] =
    pair<long int,patString>(10000,patString("")) ;

  integerValues["BTRUnfeasibleCGIterations"] =
    pair<long int,patString>(0,patString("")) ;

  integerValues["BTRQuasiNewtonUpdate"] =
    pair<long int,patString>(1,patString("")) ;

  integerValues["BTRUsePreconditioner"] =
    pair<long int,patString>(1,patString(" If 1, the subproblem is  preconditioned using a modified Cholesky factorization (see Schnabel and Eskow, 1991).")) ;


  integerValues["solvoptMaxIter"] =
    pair<long int,patString>(1000,patString(" Maximum number of iterations for algorithm SOLVOPT")) ;

  integerValues["solvoptDisplay"] =
    pair<long int,patString>(1,patString(" Controls the display of the algorithm. See <a href='http://www.kfunigraz.ac.at/imawww/kuntsevich/solvopt/ps/manual.pdf'>SOLVOPT manual</a>. ")) ;

  // integerValues["donlp2nreset"] =
  //   pair<long int,patString>(9,patString("See DONLP2 manual. In general, it should not be changed.")) ;

  integerValues["biogemeDisplay"] = 
    pair<long int,patString>(0,patString("If 0, only general messages are displayed on the screen. If 1, more detailed messages are provided. If 2 or more, messages designed for debugging purposes are also included.")) ;

  integerValues["biogemeLogDisplay"] = 
    pair<long int,patString>(1,patString("If 0, only general messages are included in the log file. If 1, more detailed messages are provided. If 2 or more, messages designed for debugging purposes are also included.")) ;

  integerValues["simulateReportForEveryObservation"] = 
    pair<long int,patString>(1,patString("If 1, the output file fir simulation includes records for each data set in the data file. If 0, only aggregate values are provided.")) ;

  realValues["NormalTruncation"] =
    pair<patReal,patString>(1.96,patString("When calculating Monte-Carlo simulation using a symmetric truncated normal distrubtion, this parameters T sets the value of the truncation in the interval [-T:T].")) ;


  realValues["MetropolisHastingsNeighborhoodSize"] =
    pair<patReal,patString>(1.0,patString("The Markov chain for the Metropolis-Hastings algorithm updates the parameters beta using the following formula: beta + rho ksi, where ksi is a draw from a normal(0,1) and rho is this parameter.")) ;

  realValues["sensitivityAnalysisAlpha"] = 
    pair<patReal,patString>(0.05,patString("Sensitivity analysis is performed by simulation. Biogeme reports the alpha and the 1-alpha quantiles of the simulated values.")) ;

  realValues["cfsqpEps"] =
    pair<patReal,patString>(6.05545e-06,
			    patString(" See CFSQP manual. In  general, it should not be changed."));

  realValues["cfsqpEpsEqn"] =
    pair<patReal,patString>(6.05545e-06,
			    patString("See CFSQP manual. In general, it should not be changed."));

  realValues["cfsqpUdelta"] =
    pair<patReal,patString>(0.0,
			    patString("See CFSQP manual. In general, it should not be changed."));

  realValues["BTRInitRadius"] =
    pair<patReal,patString>(1.0,
			    patString("Defines the initial radius of the trust region in algorithm BIO."));

  realValues["tTestThreshold"] =
    pair<patReal,patString>(1.96,
			    patString("Set the threshold for the <em>t</em>-test hypothesis tests. If the absolute value of a <em>t</em>-test is less than the value of this parameter, a symbol * will be appended to the relevant line in the report file (or the symbol defined by the previous parameter)."));

  realValues["singularValueThreshold"] =
    pair<patReal,patString>(1.0e-4,
			    patString("A singular value lesser of equal to this value is considered to be zero."));

  integerValues["detectMissingValue"] =
    pair<patULong,patString>(0,
			     patString("This parameter is used mainly for debugging purposes. If set to 1, any time that a literal takes the values given by the parameter 'missingValue', an error message is triggered."));

  realValues["missingValue"] =
    pair<patReal,patString>(99999.0,
			    patString("Default value for composite literals."));

  realValues["BTREta1"] =
    pair<patReal,patString>(0.01,
			    patString(""));

  realValues["BTREta2"] =
    pair<patReal,patString>(0.9,
			    patString(""));

  realValues["BTRGamma1"] =
    pair<patReal,patString>(0.5,
			    patString(""));

  realValues["BTRGamma2"] =
    pair<patReal,patString>(0.5,
			    patString(""));

  realValues["BTRIncreaseTRRadius"] =
    pair<patReal,patString>(2,
			    patString(" Defines the factor by which the radius of the trust region is multiplied after a successful iteration.")); 

  realValues["BTRMaxTRRadius"] =
    pair<patReal,patString>(1.0e10,
			    patString(" Defines the maximum radius of the trust region. If this radius is reached, the trust region is not enlarged anymore."));

  realValues["BTRTypf"] =
    pair<patReal,patString>(1.0,
			    patString("Typical value of the log likelihood function, with opposite sign "));

  realValues["BTRTolerance"] =
    pair<patReal,patString>(6.05545e-06,
			    patString("Value used for the stopping criterion of the BIO algorithm."));

  realValues["TolSchnabelEskow"] =
    pair<patReal,patString>(0.00492157,
			    patString(""));

  realValues["BTRInitRadius"] =
    pair<patReal,patString>(1.0,
			    patString("Defines the initial radius of the trust region."));

  realValues["BTRMinTRRadius"] =
    pair<patReal,patString>(1.0e-7,
			    patString(" Defines the minimum radius of the trust region. If this radius is reached, the iterations are interrupted."));

  realValues["BTRArmijoBeta1"] =
    pair<patReal,patString>(0.1,
			    patString(""));

  realValues["BTRArmijoBeta2"] =
    pair<patReal,patString>(0.9,
			    patString(""));

  realValues["TSFractionGradientRequired"] =
    pair<patReal,patString>(0.1,
			    patString(""));

  realValues["TSExpTheta"] =
    pair<patReal,patString>(0.5,
			    patString(""));

  realValues["BTRKappaUbs"] =
    pair<patReal,patString>(0.1,
			    patString(""));

  realValues["BTRKappaLbs"] =
    pair<patReal,patString>(0.9,
			    patString(""));

  realValues["BTRKappaFrd"] =
    pair<patReal,patString>(0.5,
			    patString(""));

  realValues["BTRKappaEpp"] =
    pair<patReal,patString>(0.25,
			    patString(""));

  realValues["solvoptErrorArgument"] =
    pair<patReal,patString>(1.0e-4,
			    patString(""));

  realValues["solvoptErrorFunction"] =
    pair<patReal,patString>(1.0e-6,
			    patString(" See <a href='http://www.kfunigraz.ac.at/imawww/kuntsevich/solvopt/ps/manual.pdf'>SOLVOPT manual</a>.  In general, it should not be changed."));

  // realValues["donlp2epsx"] =
  //   pair<patReal,patString>(1.0e-5,
  // 			    patString("See DONLP2 manual. In general, it should not be changed."));

  // realValues["donlp2delmin"] =
  //   pair<patReal,patString>(1.0e-6,
  // 			    patString("See DONLP2 manual. In general, it should not be changed."));

  // realValues["donlp2smallw"] =
  //   pair<patReal,patString>(3.66685e-11,
  // 			    patString("See DONLP2 manual. In general, it should not be changed."));

  // realValues["donlp2epsdif"] =
  //   pair<patReal,patString>(0.0,
  // 			    patString("See DONLP2 manual. In general, it should not be changed."));

  // IPOPT
  realValues["IPOPTtol"] = 
    pair<patReal,patString>(1.e-6,
   			    patString("IPOPT: Desired convergence tolerance (relative). See IPOPT documentation"));
  
  integerValues["IPOPTmax_iter"] = 
    pair<patULong,patString>(3000,
			     patString("IPOPT: Maximum number of iterations. See IPOPT documentation"));

  realValues["IPOPTmax_cpu_time"] = 
    pair<patReal,patString>(1.e6,
   			    patString("IPOPT: Maximum number of CPU seconds. See IPOPT documentation"));
  
  realValues["IPOPTacceptable_tol"] = 
    pair<patReal,patString>(1.e-4,
   			    patString("IPOPT: \"Acceptable\" convergence tolerance (relative). See IPOPT documentation"));

  integerValues["IPOPTacceptable_iter"] = 
    pair<patULong,patString>(5,
			     patString("IPOPT: Number of \"acceptable\" iterates before triggering termination. See IPOPT documentation"));

  ofstream htmlFile(docFile.c_str()) ;
  htmlFile << printDocumentation() ;
  htmlFile.close() ;
  patOutputFiles::the()->addUsefulFile(docFile,"Documentation of the parameters") ;
  GENERAL_MESSAGE("Parameter documentation generated: " << docFile) ;

}


patReal bioParameters::getValueReal(patString p, patError*& err) const {
  map<patString,pair<patReal,patString> >::const_iterator found = realValues.find(p) ;
  if (found == realValues.end()) {
    stringstream str ;
    str << "Unknown parameter: " << p ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  else {
    patReal x = found->second.first ;
    return x ;
  }
  return patReal() ;
}


patReal bioParameters::getValueReal(patString p) const {
  map<patString,pair<patReal,patString> >::const_iterator found = realValues.find(p) ;
  if (found == realValues.end()) {
    return patReal() ;
  }
  else {
    return found->second.first ;
  }
}

patString bioParameters::getValueString(patString p, patError*& err) const {
  map<patString,pair<patString,patString> >::const_iterator found = stringValues.find(p) ;
  if (found == stringValues.end()) {
    stringstream str ;
    str << "Unknown parameter: " << p ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patString() ;
  }
  else {
    return found->second.first ;
  }

}

patString bioParameters::getValueString(patString p) const {
  map<patString,pair<patString,patString> >::const_iterator found = stringValues.find(p) ;
  if (found == stringValues.end()) {
    return patString() ;
  }
  else {
    return found->second.first ;
  }

}

long bioParameters::getValueInt(patString p, patError*& err) const {
  map<patString,pair<long,patString> >::const_iterator found = integerValues.find(p) ;
  if (found == integerValues.end()) {
    stringstream str ;
    str << "Unknown parameter: " << p ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return long() ;
  }
  else {
    return found->second.first ;
  }
}

long bioParameters::getValueInt(patString p) const {
  map<patString,pair<long,patString> >::const_iterator found = integerValues.find(p) ;
  if (found == integerValues.end()) {
    return long() ;
  }
  else {
    return found->second.first ;
  }
}

void bioParameters::setValueReal(patString p, patReal v, patError*& err) {
  map<patString,pair<patReal,patString> >::iterator found = realValues.find(p) ;
  if (found == realValues.end()) {
    stringstream str ;
    str << "Paramater " << p << " is unknown." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  realValues[p] = pair<patReal,patString>(v,found->second.second) ;
}

void bioParameters::setValueString(patString p, patString v, patError*& err) {
  map<patString,pair<patString,patString> >::iterator found = stringValues.find(p) ;
  if (found == stringValues.end()) {
    stringstream str ;
    str << "Paramater " << p << " is unknown." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  stringValues[p] = pair<patString,patString>(v,found->second.second) ;
}

void bioParameters::setValueInt(patString p, long v, patError*& err) {
  map<patString,pair<long,patString> >::iterator found = integerValues.find(p) ;
  if (found == integerValues.end()) {
    stringstream str ;
    str << "Paramater " << p << " is unknown." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  integerValues[p] = pair<long,patString>(v,found->second.second) ;
}

void bioParameters::readParamFile(patString aFile, patError*& err) {

  err = new patErrMiscError("Not yet implemented") ;
  WARNING(err->describe()) ;
  return ;
  
}


patBoolean bioParameters::isRealParam(patString paramName) const {
  map<patString,pair<patReal,patString> >::const_iterator found ;
  found = realValues.find(paramName) ;
  return (found != realValues.end()) ;
}
patBoolean bioParameters::isStringParam(patString paramName) const {
  map<patString,pair<patString,patString> >::const_iterator found ;
  found = stringValues.find(paramName) ;
  return (found != stringValues.end()) ;
  
}
patBoolean bioParameters::isIntParam(patString paramName) const {
  map<patString,pair<long int,patString> >::const_iterator found ;
  found = integerValues.find(paramName) ;
  return (found != integerValues.end()) ;
}

void bioParameters::setParameters(map<patString, patString>* aDict, 
				  patError*& err) {
  if (aDict == NULL) {
    err = new patErrNullPointer("map<patString, patString>") ;
    WARNING(err->describe()) ;
    return ;
  }
  vector<patString> unknown ;
  for (map<patString, patString>::iterator i = aDict->begin() ;
       i != aDict->end() ;
       ++i) {
    patBoolean known(patFALSE) ;
    if (isRealParam(i->first)) {
      known = patTRUE ;
      patReal r ;
      if (!from_string<patReal>(r,i->second,std::dec)) {
	stringstream str ;
	str << "Error in converting the following string: " << i->second ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      setValueReal(i->first,r,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    if (isStringParam(i->first)) {
      known = patTRUE ;
      setValueString(i->first,i->second,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    if (isIntParam(i->first)) {
      known = patTRUE ;
      long int r ;
      if (!from_string<long int>(r,i->second,std::dec)) {
	stringstream str ;
	str << "Error in converting the following string: " << i->second ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe()) ;
	return ;
      }
      setValueInt(i->first,r,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
    }
    if (!known) {
      unknown.push_back(i->first) ;
    }
    
  }
  if (unknown.size() > 0) {
    stringstream str ;
    if (unknown.size() == 1) {
      str << "Unknown parameter: " << unknown[0] ;
    }
    else {
      str << "Unknown parameters: " ;
      for (patULong i = 0 ; i < unknown.size() ; ++i) {
	if (i != 0) {
	  str << ", " ;
	}
	str << unknown[i] ;
      }
    }
    str << ". The list of parameters can be found in the file __parametersUsed.py." ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }

  return ;
}

patString bioParameters::printPythonCode() const {
  stringstream str ;
  for (map<patString,pair<patReal,patString> >::const_iterator i = realValues.begin() ;
       i != realValues.end() ;
       ++i) {
    str << "BIOGEME_OBJECT.PARAMETERS['"<<i->first<<"'] = \""<<i->second.first<<"\"" << endl ;
  }
  for (map<patString,pair<patString,patString> >::const_iterator i = stringValues.begin() ;
       i != stringValues.end() ;
       ++i) {
    str << "BIOGEME_OBJECT.PARAMETERS['"<<i->first<<"'] = \""<<i->second.first<<"\"" << endl ;
  }
  for (map<patString,pair<long int,patString> >::const_iterator i = integerValues.begin() ;
       i != integerValues.end() ;
       ++i) {
    str << "BIOGEME_OBJECT.PARAMETERS['"<<i->first<<"'] = \"" << i->second.first <<"\"" << endl ;
  }
  return patString(str.str()) ;


}

trParameters bioParameters::getTrParameters(patError*& err) const {

  trParameters p ;
  p.eta1 = getValueReal("BTREta1",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.eta2 = getValueReal("BTREta2",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.gamma2 = getValueReal("BTRGamma2",err);
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.beta = getValueReal("BTRIncreaseTRRadius",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.maxIter = getValueInt("BTRMaxIter",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.initQuasiNewtonWithTrueHessian = 
    getValueInt("BTRInitQuasiNewtonWithTrueHessian",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.initQuasiNewtonWithBHHH = 
    getValueInt("BTRInitQuasiNewtonWithBHHH",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.significantDigits = getValueInt("BTRSignificantDigits",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.usePreconditioner = getValueInt("BTRUsePreconditioner",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.maxTrustRegionRadius = getValueReal("BTRMaxTRRadius",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.typicalF = getValueReal("BTRTypf",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.tolerance = getValueReal("BTRTolerance",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.toleranceSchnabelEskow = getValueReal("TolSchnabelEskow",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.exactHessian = getValueInt("BTRExactHessian",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.cheapHessian = getValueInt("BTRCheapHessian",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.initRadius = getValueReal("BTRInitRadius",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.minRadius = getValueReal("BTRMinTRRadius",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.armijoBeta1 = getValueReal("BTRArmijoBeta1",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.armijoBeta2 = getValueReal("BTRArmijoBeta2",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.stopFileName = getValueString("stopFileName",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.startDraws = getValueInt("BTRStartDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.increaseDraws = getValueInt("BTRIncreaseDraws",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.maxGcpIter = getValueInt("maxGcpIter",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.fractionGradientRequired = getValueReal("TSFractionGradientRequired",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.expTheta = getValueReal("TSExpTheta",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.infeasibleCgIter = getValueInt("BTRUnfeasibleCGIterations",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.quasiNewtonUpdate = getValueInt("BTRQuasiNewtonUpdate",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.kappaUbs = getValueReal("BTRKappaUbs",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.kappaLbs = getValueReal("BTRKappaLbs",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.kappaFrd = getValueReal("BTRKappaFrd",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }
  p.kappaEpp = getValueReal("BTRKappaEpp",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trParameters() ;
  }

  return p ;

}



patString bioParameters::printDocumentation() const {
  patAbsTime now ;
  now.setTimeOfDay() ;

  stringstream str ;

  str << "<!DOCTYPE html PUBLIC '-//W3C//DTD XHTML 1.0 Transitional//EN'" << endl ;
  str << "        'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'>" << endl ;
  str << "<html xmlns='http://www.w3.org/1999/xhtml'>" << endl ;
  str << "<head>" << endl ;
  str << "<title>Python Biogeme</title>" << endl ;

  str << "<p>" << bioVersion::the()->getVersionInfoDate() << "</p>" << endl ;
  str << "<p>Home page: <a href='" << bioVersion::the()->getUrl() << "' target='_blank'>"<<  bioVersion::the()->getUrl() <<"</a></p>" << endl ;
  str << "<p>Submit questions to <a href='" << bioVersion::the()->getUrlUsersGroup() << "' target='_blank'>"<<  bioVersion::the()->getUrlUsersGroup() <<"</a></p>" << endl ;

  str << "<p>" << bioVersion::the()->getVersionInfoAuthor() << "</p>" << endl ;
  str << "<p>This file has automatically been generated on" << endl ;
  str <<  now.getTimeString(patTsfFULL) << "</p>" << endl ;

  
  str << "<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />" << endl ;
  str << "</head>" << endl ;
  str << "<body>" << endl ;
  str << "<h1>Python Biogeme Parameters</h1>" << endl ;
  str << "" << endl ;
  str << "<p>The execution of Biogeme can be controlled by user defined parameters. The syntax is" << endl ;
  str << "<code>" << endl ;
  str << "<pre>" << endl ;
  str << "BIOGEME_OBJECT.PARAMETERS['parameterName'] = \"value\"" << endl ;
  str << "</pre>" << endl ;
  str << "</code>" << endl ;
  str << "</p>" << endl ;
  str << "" << endl ;
  str << "<p>" << endl ;
  str << "The <code>parameterName</code> must be one of the names listed below. Depending on the nature of the parameter, the value must be an integer, a real number or a string of characters. The value must be delimited by double quotes." << endl ;
  str << "</p>" << endl ;
  str << "" << endl ;
  str << "<table>" << endl ;
  str << "<tr><th>Parameter name</th>" << endl ;
  str << "<th>Type of value</th>" << endl ;
  str << "<th>Default value</th>" << endl ;
  str << "<th>Description</th>" << endl ;
  str << "</tr>" << endl ;
  

  for (map<patString,pair<patReal,patString> >::const_iterator i = realValues.begin() ;
       i != realValues.end() ;
       ++i) {
    str << "<tr valign='top'>" << endl ;
    str << "<td><code>" << i->first  <<"</code></td>" << endl ;
    str << "<td>Real</td>" << endl ;
    str << "<td><code>" << i->second.first << " </code></td>" << endl ;
    str << "<td>"<< i->second.second << "</td>" << endl ;
    str << "</tr>" << endl ;

  }
  for (map<patString,pair<patString,patString> >::const_iterator i = stringValues.begin() ;
       i != stringValues.end() ;
       ++i) {
    str << "<tr valign='top'>" << endl ;
    str << "<td><code>" << i->first  <<"</code></td>" << endl ;
    str << "<td>String</td>" << endl ;
    str << "<td><code>" << i->second.first << " </code></td>" << endl ;
    str << "<td>"<< i->second.second << "</td>" << endl ;
    str << "</tr>" << endl ;
  }
  for (map<patString,pair<long int,patString> >::const_iterator i = integerValues.begin() ;
       i != integerValues.end() ;
       ++i) {
    str << "<tr valign='top'>" << endl ;
    str << "<td><code>" << i->first  <<"</code></td>" << endl ;
    str << "<td>Integer</td>" << endl ;
    str << "<td><code>" << i->second.first << " </code></td>" << endl ;
    str << "<td>"<< i->second.second << "</td>" << endl ;
    str << "</tr>" << endl ;
  }

  str << "</table>" << endl ;
  str << "</body>" << endl ;
  str << "</html>" << endl ;

  return patString(str.str()) ;


}


bioParameterIterator<patReal>* bioParameters::createRealIterator() {
  return( new bioParameterIterator<patReal>(&realValues)) ;
}

bioParameterIterator<long int>* bioParameters::createIntegerIterator() {
  return( new bioParameterIterator<long int>(&integerValues)) ;
}

bioParameterIterator<patString>* bioParameters::createStringIterator() {
  return( new bioParameterIterator<patString>(&stringValues)) ;
}

patString bioParameters::getDocumentationFilename() const {
  return docFile ;
}
