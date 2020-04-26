//-*-c++-*------------------------------------------------------------
//
// File name : trLineSearchAlgo.cc
// Author :    Michel Bierlaire
// Date :      Tue Nov 16 10:26:01 2004
//
//--------------------------------------------------------------------


#include <iomanip>
#include <sstream>
#include <numeric>
#include "patConst.h"
#include "patMath.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "trLineSearchAlgo.h"
#include "trPrecond.h"
#include "patNonLinearProblem.h"
#include "trFunction.h"
#include "trHessian.h"
#include "trBFGS.h"
#include "trSR1.h"
#include "trPrecondCG.h"
#include "patFileExists.h"
#include "trParameters.h"

trLineSearchAlgo::trLineSearchAlgo(patNonLinearProblem* aProblem,
				   const trVector& initSolution,
				   trParameters tp,
				   patError*& err) :
  trNonLinearAlgo(aProblem),
  solution(initSolution),
  theParameters(tp),
  hessian(NULL),
  iterInfo(NULL),
  iter(0) {
  
  if (theProblem == NULL) {
    err = new patErrNullPointer("patNonLinearProblem") ;
    WARNING(err->describe()) ;
    return ;
  }

  f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  if (f == NULL) {
    err = new patErrNullPointer("trFunction") ;
    WARNING(err->describe()) ;
    return ;
  }

  // Allocate memory for the hessian or its approximation

  hessian = new trHessian(theParameters,f->getDimension()) ;

  if (theParameters.armijoBeta2 <= theParameters.armijoBeta1) {
    stringstream str ;
    str << "Second armijo parameter (" << theParameters.armijoBeta2 <<") must be larger than the first (" << theParameters.armijoBeta1 << ")" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return ;
  }
  
}

trLineSearchAlgo::~trLineSearchAlgo() {

  if (hessian != NULL) {
    DELETE_PTR(hessian) ;
  }
}


trVector trLineSearchAlgo::getSolution(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector();
  }
  return(solution) ;
}

patReal trLineSearchAlgo::getValueSolution(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (f == NULL) {
    err = new patErrNullPointer("patFunction") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }
  patBoolean success ;
  patReal result = f->computeFunction(&solution,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!success) {
    WARNING("Numerical problem in evaluating the function at the solution") ;
  }
  
  
  return result ;
}

patString trLineSearchAlgo::run(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return (err->describe());
  }

  // Parameters

  // Compute the function and gradient at the initial solution

  //  DEBUG_MESSAGE("Compute the function and gradient at the initial solution") ;

  if (f == NULL) {
    err = new patErrNullPointer("trFunction") ;
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }
  
  if (!f->isGradientAvailable()) {
    err = new patErrMiscError("Gradient is unavailable") ;
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }
  
  gk.resize(f->getDimension()) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe());
  }
  gradientCandidate.resize(gk.size()) ;

  iter = 0 ;
  GENERAL_MESSAGE("    gmax Iter    f(x)     Step") ; 

  patBoolean success ;
  function = f->computeFunctionAndDerivatives(&solution,&gk,NULL,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe());
  }
  
  if (!success) {
    err = new patErrMiscError("Numerical problem in function and gradient evaluation") ;
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }
  
  
  
  patVariables gradientCandidate(gk.size()) ;

  patBoolean goodStep(patTRUE) ;

  patReal gMax ;

  while (!stop(gMax,err) && iter < theParameters.maxIter && goodStep) {
	  
    ++iter ;
    
    hessian = f->computeCheapHessian(hessian,err) ;
  
    if (err != NULL) {
      WARNING(err->describe()) ;
      return NULL;
    }
    if (!success) {
      err = new patErrMiscError("Numerical problem in hessian computation") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    patHybridMatrix* hessianMatrix = hessian->getHybridMatrixPtr() ;
    hessianMatrix->cholesky(theParameters.toleranceSchnabelEskow,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
    patVariables direction = hessianMatrix->solve(-gk,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }

    DEBUG_MESSAGE("====================") ;
    DEBUG_MESSAGE("Direction = " << direction) ;
    DEBUG_MESSAGE("====================") ;
    patReal slope = inner_product(gk.begin(),gk.end(),direction.begin(),0.0) ;

    DEBUG_MESSAGE("slope = " << slope) ;
    for (patReal alpha = 0 ; alpha <= .1 ; alpha += 0.01) {
      patVariables debugvec(gk.size()) ;
      patVariables xdeb = solution + alpha * direction ;
      patReal debugval = f->computeFunctionAndDerivatives(&xdeb,	
							  &debugvec,
							  NULL,
							  &success,
							  err) ;
      DEBUG_MESSAGE(alpha << '\t' << debugval) ;
    }
    
    patReal step = 1.0  ;
    
    goodStep = patFALSE ;
    
    patReal lowerBound(0.0) ;
    patReal infinity = 10.0e15 ;
    patReal upperBound(infinity) ;
    while (!goodStep && (upperBound - lowerBound >= 1.0e-7)) {

      DELETE_PTR(iterInfo) ;
      iterInfo = new stringstream ;
      
      
      *iterInfo << setfill(' ') << setiosflags(ios::scientific|ios::showpos) 
		<< setprecision(2) << gMax << " " ;
      
      *iterInfo << resetiosflags(ios::showpos) ;
      
      *iterInfo << setw(4) << iter << " "  ;
      
      patVariables candidate = solution + step * direction ;

      DEBUG_MESSAGE("candidate = " << candidate) ;

      valCandidate = f->computeFunctionAndDerivatives(&candidate,
						      &gradientCandidate,
						      NULL,
						      &success,
						      err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return(err->describe());
      }
      
      if (!success) {
	err = new patErrMiscError("Numerical problem in function and gradient evaluation") ;
	WARNING(err->describe()) ;
	return (err->describe()) ;
      }


      *iterInfo <<  setfill(' ') 
		<< setiosflags(ios::scientific|ios::showpos) 
		<< setprecision(7) 
		<< function << " " ; 
      
      *iterInfo <<  setfill(' ') 
		<< setiosflags(ios::scientific|ios::showpos) 
		<< setprecision(7) 
		<< step << " " ; 
      
      // Check the first Armijo condition
      
      DEBUG_MESSAGE("valCandidate = " << valCandidate) ;
      DEBUG_MESSAGE("function =     " << function) ;
      DEBUG_MESSAGE("step =         " << step) ;
      DEBUG_MESSAGE("slope =        " << slope) ;
      DEBUG_MESSAGE("Threshold =    " << function + step * theParameters.armijoBeta1 * slope) ;
      if (valCandidate <= function + step * theParameters.armijoBeta1 * slope) {
	// Check the second  Armijo condition
	
	patReal newSlope = inner_product(gradientCandidate.begin(),
					 gradientCandidate.end(),
					 direction.begin(),
					 0.0) ;
	
	if (newSlope >= theParameters.armijoBeta2 * slope) {
	  goodStep = patTRUE ;
	  solution = candidate ;
	  function = valCandidate ;
	  gk = gradientCandidate ;
	}
	else {
	  *iterInfo << "too short " ;
	    // Step too short
	  lowerBound = step  ;
	  if (upperBound < infinity) {
	    step = (lowerBound + upperBound) / 2.0 ;
	  }
	  else {
	    step *= 2.0 ;
	  }
	}
      }
      else {
	// Step too long
	*iterInfo << "too long " ;
	upperBound = step ;
	step = (lowerBound + upperBound) / 2.0 ;
      }
      GENERAL_MESSAGE(iterInfo->str()) ;
    }


  }


  DEBUG_MESSAGE("End of iterations") ;
  DEBUG_MESSAGE("gMax = " << gMax) ;
  DEBUG_MESSAGE("iter=" << iter << " maxIter = " << theParameters.maxIter) ;
  DEBUG_MESSAGE("goodStep = " << goodStep) ;
  
  cout << setprecision(7) << endl  ;
  
  if (iter == theParameters.maxIter) {
    GENERAL_MESSAGE("Maximum number of iterations reached") ;
    return patString("Maximum number of iterations reached") ;
  }
  else if (!goodStep) {
    GENERAL_MESSAGE("Unable to find an acceptable step into current direction") ;
    return patString("Radius of the trust region is too small") ;
  }
  else {
    GENERAL_MESSAGE("Convergence reached...") ;
    
    DETAILED_MESSAGE("Solution = " << solution) ;
    DETAILED_MESSAGE("gk=" << gk) ;
    return patString("Convergence reached...") ;
  }
  return patString("Unknown termination status") ;
}


patBoolean trLineSearchAlgo::stop(patReal& gMax,patError*& err) {

  if (patFileExists()(theParameters.stopFileName)) {
    WARNING("Iterations interrupted by the user with the file " 
    	    << theParameters.stopFileName) ;
    return patTRUE ;
  }

  return checkOpt(solution,gk,gMax,err) ;


}

patBoolean trLineSearchAlgo::checkOpt(const trVector& x,
					const trVector& g,
					patReal& gMax,
					patError*& err) {
  
  gMax = 0.0 ;

  trVector::const_iterator gIter = g.begin() ;
  trVector::const_iterator xIter = solution.begin() ;
  for ( ; gIter != g.end() ; ++gIter, ++xIter) {
    patReal gRel = patAbs(*gIter) * patMax(1.0,patAbs(*xIter)) / 
      patMax(patAbs(function),theParameters.typicalF) ;
    gMax = patMax(gMax,gRel) ;
  }

  DEBUG_MESSAGE("gMax = " <<gMax) ;
  return (gMax < theParameters.tolerance) ;
}

patString trLineSearchAlgo::getName() {
  return patString("Simple line search algorithm") ;
}

patVariables trLineSearchAlgo::getLowerBoundsLambda() {
  return patVariables(gk.size()) ;
}

patVariables trLineSearchAlgo::getUpperBoundsLambda() {
  return patVariables(gk.size()) ;

}
void trLineSearchAlgo::defineStartingPoint(const patVariables& x0) {
  solution = x0 ;
}

patULong trLineSearchAlgo::nbrIter() {
  return iter ;
}
