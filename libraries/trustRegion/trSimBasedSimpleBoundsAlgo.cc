//-*-c++-*------------------------------------------------------------
//
// File name : trSimBasedSimpleBoundsAlgo.cc
// Author :    Michel Bierlaire
// Date :      Fri Jun 11 09:39:31 2004
//
// Implementation of algorithm 12.1.1: trust-Region Algorithm for simple bounds
// The number of draws for Monte Carlo simulation of the objective function 
// is controlled by the algorithm
//
// Source: Conn, Gould Toint (2000) Trust Region Methods
//--------------------------------------------------------------------


#include <iomanip>
#include "patConst.h"
#include "patMath.h"
#include "trParameters.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "trSimBasedSimpleBoundsAlgo.h"
#include "trPrecond.h"
#include "patNonLinearProblem.h"
#include "patSimBasedOptimizationProblem.h"
#include "trFunction.h"
#include "trBounds.h"
#include "trHessian.h"
#include "trBFGS.h"
#include "trSR1.h"
#include "trPrecondCG.h"
#include "trBoxGCP.h"
#include "patFileExists.h"
#include "patIterationBackup.h"

trSimBasedSimpleBoundsAlgo::trSimBasedSimpleBoundsAlgo(patSimBasedOptimizationProblem* aProblem,
						       const trVector& initSolution,
						       trParameters theParameters,
						       patError*& err) :
  trNonLinearAlgo(aProblem->theProblem),
  theSimBasedProblem(aProblem),
  status(trUNKNOWN),
  solution(initSolution),
  radius(theParameters.initRadius),
  trueHessian(NULL),
  quasiNewton(NULL),
  precond(NULL),
  iterInfo(NULL),
  mustInitBFGS(patTRUE),
  iter(0) {


  userNbrDraws = theSimBasedProblem->getUserNbrOfDraws() ;
  
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

  bounds = new trBounds(theProblem->getLowerBounds(err),
			theProblem->getUpperBounds(err)) ;


  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patBoolean feas = bounds->isFeasible(initSolution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (!feas) {
    WARNING("Initial solution not feasible. Its projection onto the feasible set is sued instead.") ;
    solution = bounds->getProjection(initSolution,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
  }

  // Allocate memory for the hessian or its approximation

  exactHessian = 
    f->isHessianAvailable() && 
    (theParameters.exactHessian != 0);
  
  if (exactHessian) {
    trueHessian = new trHessian(theParameters,
				f->getDimension()) ;
  }
  
   cheapHessian = 
     !exactHessian && f->isCheapHessianAvailable() && 
     (theParameters.cheapHessian != 0) ;
   
   if (cheapHessian) {
     trueHessian = new trHessian(theParameters,
				 f->getDimension()) ;
   }
  

   // DEBUG_MESSAGE("trSimBasedSimpleBoundsAlgo created") ;

}

trSimBasedSimpleBoundsAlgo::~trSimBasedSimpleBoundsAlgo() {

  if (bounds != NULL) {
    DELETE_PTR(bounds) ;
  }
  if (trueHessian != NULL) {
    DELETE_PTR(trueHessian) ;
  }
  if (quasiNewton != NULL) {
    DELETE_PTR(quasiNewton) ;
  }
  if (precond != NULL) {
    DELETE_PTR(precond) ;
  }
}

trSimBasedSimpleBoundsAlgo::trTermStatus trSimBasedSimpleBoundsAlgo::getTermStatus() {
  return status ;
}

trVector trSimBasedSimpleBoundsAlgo::getSolution(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector();
  }
  return(solution) ;
}

patReal trSimBasedSimpleBoundsAlgo::getValueSolution(patError*& err) {
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

patString trSimBasedSimpleBoundsAlgo::run(patError*& err) {
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
  lowerLambda.resize(gk.size(),0.0) ;
  upperLambda.resize(gk.size(),0.0) ;

  patBoolean success ;
  function = f->computeFunctionAndDerivatives(&solution,&gk,trueHessian,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe());
  }
  
  if (!success) {
    err = new patErrMiscError("Numerical problem in function and gradient evaluation at the starting point") ;
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }

  //  DEBUG_MESSAGE("f(x0)=" << function) ;

  trVector dummy ;
  hessian = trueHessian ;
  
  iter = 0 ;
  GENERAL_MESSAGE("Draws            gmax ratio Iter   radius        f(x)     Status           rhok nFree") ; 
  
  
  patBoolean precondUsed = patFALSE ;
  
  patReal gMax ;
  
  // ***** START ITERATIONS
      
  unsigned long startDraws = theParameters.startDraws ;
  unsigned long algoNbrDraws = (userNbrDraws < startDraws) 
    ? userNbrDraws : startDraws ;  
  theSimBasedProblem->setAlgoNbrOfDraws(algoNbrDraws) ;


  while (!(stop(gMax,err) && (algoNbrDraws == userNbrDraws)) &&
	 iter < theParameters.maxIter && radius >= theParameters.minRadius) {

    if (theBackup != NULL) {
      theBackup->saveCurrentIteration() ;
    }
    if (patFileExists()(theParameters.stopFileName)) {
      WARNING("Iterations interrupted by the user with the file " 
	      << theParameters.stopFileName) ;
      return("Iterations interrupted by the user") ;
    }
  

//     DEBUG_MESSAGE("gmax   = " << gMax) ;
//     DEBUG_MESSAGE("target = " <<  patParameters::the()->getBTRTolerance()) ;
    patReal optimalityRatio = 
      log(gMax) / log(theParameters.tolerance) ;
    patReal drawRatio = patReal(algoNbrDraws) / patReal(userNbrDraws) ;
    unsigned short percent = (optimalityRatio > 0) 
      ? unsigned(100.0 * optimalityRatio / drawRatio ) 
      : 0 ;
    if (percent > 100) {
      percent = 100 ;
    }

    DELETE_PTR(iterInfo) ;
    iterInfo = new stringstream ;
    
    //    patBoolean continueIteration = patTRUE ;
    
    *iterInfo << setfill(' ') << setw(5) << theSimBasedProblem->getAlgoNbrOfDraws() << "/"  << setfill(' ') << setw(5) << theSimBasedProblem->getUserNbrOfDraws() << " " ;

//     DEBUG_MESSAGE("OPTIM " << optimalityRatio) ;
//     DEBUG_MESSAGE("DRAW  " << drawRatio) ;
    *iterInfo << setfill(' ') << setiosflags(ios::scientific|ios::showpos) 
	      << setprecision(2) << gMax << " " << setw(4) << percent << "% " ;
    
    ++iter ;
    *iterInfo << resetiosflags(ios::showpos) ;
    
    *iterInfo << setw(4) << iter << " " << setfill(' ') 
	      << setiosflags(ios::scientific) << setprecision(2) 
	      << radius << " " ;
    *iterInfo <<  setfill(' ') 
	      << setiosflags(ios::scientific|ios::showpos) 
	      << setprecision(7) 
	      << function << " " ; 
    
    if (optimalityRatio >= drawRatio) {
      radius = theParameters.initRadius ;
      algoNbrDraws *= theParameters.increaseDraws ;
      if (algoNbrDraws >= userNbrDraws) {
	algoNbrDraws = userNbrDraws ;
      }
      theSimBasedProblem->setAlgoNbrOfDraws(algoNbrDraws) ;
      function = f->computeFunctionAndDerivatives(&solution,&gk,trueHessian,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return(err->describe());
      }
      
      if (!success) {
	err = new patErrMiscError("Numerical problem in function and gradient evaluation at the starting point") ;
	WARNING(err->describe()) ;
	return (err->describe()) ;
      }
      
      //  DEBUG_MESSAGE("f(x0)=" << function) ;
      
      trVector dummy ;
      hessian = trueHessian ;
    }
    // Compute the Generalized Cauchy Point
    
    //     DEBUG_MESSAGE("solution=" << solution) ;
    //     DEBUG_MESSAGE("gk=" << gk) ;
    trVector direction(-gk) ;
    
    //     DEBUG_MESSAGE("alpha\tx\tf") ;
    //     for (patReal alpha = 0 ; alpha <= 0.05 ; alpha += 0.001) {
    //       trVector tmp = solution + alpha * direction ;
    //       for (unsigned int i = 0 ; i < tmp.size() ; ++i) {
    // 	if (tmp[i] < bounds->getLower(i,err)) {
    // 	  tmp[i] = bounds->getLower(i,err) ;
    // 	}
    // 	if (tmp[i] > bounds->getUpper(i,err)) {
    // 	  tmp[i] = bounds->getUpper(i,err) ;
    // 	}
    //       }
    //       patReal tmpResult = f->getFunction(&tmp,&success,err) ;
    //       DEBUG_MESSAGE(alpha << '\t' << tmp << '\t' << tmpResult) ;
    //     }


    trBoxGCP theGcp(*bounds, radius, solution, direction, gk, *hessian) ;
    
    trVector gcp = theGcp.computeGCP(theParameters.maxGcpIter,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
    
    patReal fgcp = f->computeFunction(&gcp,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
    //     DEBUG_MESSAGE("f = " << fgcp << " gcp = " << gcp) ;
    if (!success) {
      DEBUG_MESSAGE("Error in fct evaluation at the GCP") ;
    
    }
    
    //    patReal normStep = radius ;

    // We consider the iteration as having failed if f(gcp) >= f(xk)  
    // In this case, we do not spend time solving the problem in the subspace,
    // which is likely to be inadequate anyway
    if (fgcp < function) {
      // Solve the trust region problem in the subspace defined by the free
      // variables at the GCP, starting at the GCP.
      
      //      patBoolean successGrad ;
      
      //      trVector gradGcp = f->getGradient(gcp,&successGrad,err) ;
      //      if (err != NULL) {
      //        WARNING(err->describe()) ;
      //        return trUNKNOWN ;
      //      }
      
      //       DEBUG_MESSAGE("Solve in the subspace") ;
      
      vector<trBounds::patActivityStatus> activity =
	bounds->getActivity(gcp,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return(err->describe()) ;
      }
      
      trVector reducedGradient ;
      trVector reducedIter ;
      
      for (unsigned long i = 0 ; i < gk.size() ; ++i) {
	if (activity[i] == trBounds::patFree) {
	  reducedGradient.push_back(gk[i]) ;
	  reducedIter.push_back(solution[i]) ;
	}
      }
      
      unsigned long nFree = reducedGradient.size() ;
      
      patBoolean cannotImproveInSubspace = patFALSE ;
      if (nFree == 0) {
	cannotImproveInSubspace = patTRUE ;
      }
      else {
	cannotImproveInSubspace = 
	  checkOpt(reducedIter,reducedGradient,gMax,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return(err->describe()) ;
	}
      }
      
      patReal normStep = radius ;
      patReal modelImprovement = -1.0 ;
      patReal valCandidate = function + 1.0 ;
      
      trVector candidate(solution.size()) ;
      
      if (cannotImproveInSubspace) {
	// 	DEBUG_MESSAGE("Cannot improve in subspace") ;
      }
      else {

	// 	DEBUG_MESSAGE("Try to improve in the subspace") ;
	// We try to find a better point in the subspace of free variables
	// using CG
	
	// This could be optimized. Indeed, if the true hessian is used, there
	// is no need to compute it entirely. Only the submatrix is
	// necessary. If the BFGS update is used, however, the entire matrix
	// must be updated. We'll worry about it later on...

	// 	DEBUG_MESSAGE("Get reduced hessian") ;
	
	trMatrixVector* reducedHessian = hessian->getReduced(activity,
							     err) ;
	

	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return(err->describe()) ;
	}
	
	if (precond != NULL) {
	  DELETE_PTR(precond) ;
	}
	if (reducedHessian->providesPreconditionner() &&
	    theParameters.usePreconditioner) {
	  precond = reducedHessian->createPreconditionner(err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return(err->describe()) ;
	  }
	  precondUsed = patTRUE ;
	}
	else {
	  precondUsed = patFALSE ;
	  precond = new trPrecond() ;
	}
	
	trBounds stepBounds(*bounds,solution,activity,radius,err) ;
	
	//DEBUG_MESSAGE("H=" << *quasiNewton) ;
	
	
	//        ///DEBUG
	//         cout << "\n#####################" <<endl ;
	//         cout << "alpha\tf\tmod" << endl ;
	//         patReal xtmp = f->getFunction(solution,&success,err) ;
	//         for (patReal alpha = 0 ; alpha < 0.1 ; alpha += 0.00001) {
	// 	  trVector dir = - alpha * gk ;
	// 	  for (unsigned int i = 0 ; i < gk.size() ; ++i) {
	// 	    if (activity[i] != trBounds::patFree) {
	// 	      dir[i] = 0.0 ;
	// 	    }
	// 	  }
	// 	  trVector tmp = solution + dir ;
	// 	  patReal valTmp = f->getFunction(tmp,&success,err) ;
	// 	  patReal modTmp = inner_product(gk.begin(),gk.end(),dir.begin(),0.0) + 
	// 	    0.5 * inner_product(dir.begin(),dir.end(),dir.begin(),0.0) ;
	// 	  cout << alpha << '\t' << valTmp << '\t' << modTmp << '\t' << (valTmp-xtmp)/modTmp << endl ;
	  
	//         }
	//        exit(-1) ;
	///ENDDEBUG

	  // 	DEBUG_MESSAGE("Build the model") ;

	  trPrecondCG model(reducedGradient,
			    reducedHessian,
			    stepBounds,
			    precond,
			    theParameters,
			    err) ;
	
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return(err->describe()) ;
	  }
	
	  trPrecondCG::trTermStatus status = model.run(err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return(err->describe()) ;
	  }
	
	  *iterInfo <<  setiosflags(ios::right)  <<setw(8) <<  setfill('*') 
		    <<  model.getStatusName(status) << setw(1) << " "  ;
	
	  trVector stepSubspace = model.getSolution(err) ;
	
	  // 	DEBUG_MESSAGE("step Subspace = " << stepSubspace) ;
	
	  // Compute infinity norm of the step
	  normStep = 0.0 ;
	  for (trVector::iterator k = stepSubspace.begin() ;
	       k != stepSubspace.end() ;
	       ++k) {
	    if (patAbs(*k) > normStep) {
	      normStep = patAbs(*k) ;
	    }
	  }
      
	  // New candidate
	  trVector::const_iterator iter = stepSubspace.begin() ;
	  for (unsigned long i = 0 ;
	       i < activity.size() ;
	       ++i) {
	    if (activity[i] == trBounds::patFree) {
	      if (iter == stepSubspace.end()) {
		stringstream str ;
		str << "Inconsistency with activity status." 
		    << " Subspace solution has only " 
		    << stepSubspace.size() 
		    << " elements"  ; 
		err = new patErrMiscError(str.str()) ;
		WARNING(err->describe()) ;
	      }
	      candidate[i] = solution[i] + *iter ;
	      ++iter ;
	    }
	    else {
	      candidate[i] = solution[i] ;
	    }
	  }
      
	  valCandidate = f->computeFunction(&candidate,&success,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return(err->describe()) ;
	  }
      
	  // 	DEBUG_MESSAGE("f =" << function << " Solution = " << solution);
	  // 	DEBUG_MESSAGE("f =" << valCandidate << " Candidate = " << candidate );
      
      
      
	  if (!success) {
	    valCandidate = patMaxReal ;
	  }
      
	  modelImprovement = model.getValue(err) ;
      
	  //        if (patAbs(modelImprovement) <= patEPSILON) {
	  //  	if (patParameters::the()->getBTRUsePreconditioner()) {
	  //  	  //	  WARNING("No model improvement. May be due to an inappropriate preconditioner.") ;
	  //  	  //WARNING("Preconditioning canceled...") ;
	  //  	  patParameters::the()->setBTRUsePreconditioner(patFALSE) ;
	  //  	  modelImprovement = patMaxReal ;
	  //  	}
	  //  	else {
	  
	  //  	  DEBUG_MESSAGE("Error.... check optimality") ;
	  //  	  if (stop(gMax)) {
	  //  	    return trCONV ;
	  //  	  }
	  //  	  stringstream str ;
	  //  	  str << "Model improvement too small: " << modelImprovement ;
	  //  	  err = new patErrMiscError(str.str()) ;
	  //  	  WARNING(err->describe()) ;
	  //  	  return trUNKNOWN;
	  //  	}
	  //        }
      
      }

      // Computation of rhok, See Conn, Gould, Toint p. 794.
      
      patReal rhok =  computeRhoK(function,valCandidate,modelImprovement) ;
      
      *iterInfo << setfill(' ') 
		<< setiosflags(ios::scientific|ios::showpos) 
		<< setprecision(2)
		<< rhok << " " << nFree << " ";
      
      trVector gkTemp(gk.size()) ;

      if (rhok >= theParameters.eta1) {
	// ***************************
	// Compute and check the gradient 
	//****************************
	
	f->computeFunctionAndDerivatives(&candidate,&gkTemp,NULL,&success,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return(err->describe());
	}
	if (!success) {
	  DEBUG_MESSAGE("Error in grad eval at the candidate") ;
	  rhok = -1.0 ;
	}
      }
      
      
      if (rhok >= theParameters.eta1) {
	// Successful iteration
	oldGk = gk ;
	oldSolution = solution ;
	solution = candidate ;
	function = valCandidate ;
	gk = gkTemp ;
	*iterInfo << " +" ;
	
	////////////////////////////////////////////
	// It is not necessary to compute the full hessian at each
	// iteration. Only the reduced hessian is used. This should be optimized
	// in the future.
	////////////////////////////////////////////
	
	hessian = computeHessian(oldSolution,oldGk,solution,gk,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return(err->describe()) ;
	}
	
	// Update trust region
	if (rhok >= theParameters.eta2) {
	  *iterInfo << "+" ;
	  // Very successful iteration
	  radius *= theParameters.beta ; 
	}
	else {
	  *iterInfo << " " ;
	}
      }
      if (rhok < theParameters.eta1) {
	*iterInfo << " - ";
	// Unsuccessful iteration
	if (normStep > patEPSILON) {
	  radius = normStep *  theParameters.gamma2 ;
	}
      }
    }
    else {
      *iterInfo << " - ";
      // Unsuccessful iteration
      radius *= theParameters.gamma2 ;
    }
    
    if (precondUsed) {
      *iterInfo << " P" ;
    }

    
    GENERAL_MESSAGE(iterInfo->str()) ;
    if (radius > theParameters.maxTrustRegionRadius) {
      radius = theParameters.maxTrustRegionRadius ;
    }
    
  }
  
  cout << setprecision(7) << endl  ;
  
  if (iter == theParameters.maxIter) {
    GENERAL_MESSAGE("Maximum number of iterations reached") ;
    status = trMAXITER;
    return patString("Maximum number of iterations reached") ;
  }
  else if (radius < theParameters.minRadius) {
    GENERAL_MESSAGE("Radius of the trust region is too small") ;
    status = trMINRADIUS ;
    return patString("Radius of the trust region is too small") ;
  }
  else {
    GENERAL_MESSAGE("Convergence reached...") ;
    
    DETAILED_MESSAGE("Solution = " << solution) ;
    DETAILED_MESSAGE("gk=" << gk) ;
    status = trCONV ;
    

    vector<trBounds::patActivityStatus> activity = 
      bounds->getActivity(solution,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
    for (unsigned long k = 0 ;
	 k < activity.size() ;
	 ++k) {
      if (activity[k] == trBounds::patLower) {
	lowerLambda[k] = -gk[k] ;
      }
      else if (activity[k] == trBounds::patUpper) {
	upperLambda[k] = -gk[k] ;
      }
    }
    theProblem->setLagrangeLowerBounds(lowerLambda,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
    theProblem->setLagrangeUpperBounds(upperLambda,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return(err->describe()) ;
    }
 

    return patString("Convergence reached...") ;
  }
  return patString("Unknown termination status") ;
}


patBoolean trSimBasedSimpleBoundsAlgo::stop(patReal& gMax, 
					    patError*& err) {

  trVector gproj = bounds->getProjection(solution-gk,err) - solution ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patTRUE ;
  }
  return checkOpt(solution,gproj,gMax,err) ;


}

trMatrixVector* 
trSimBasedSimpleBoundsAlgo::computeHessian( trVector& previousIterate,
				    trVector& previousGradient,
				    trVector& currentIterate,
				    trVector& currentGradient,
				   patError*& err) {

  patBoolean success ;
  if (trueHessian != NULL) {

    //    DEBUG_MESSAGE("Compute true hessian") ;
    if (!exactHessian) {

      f->computeFunctionAndDerivatives(&currentIterate,&currentGradient,NULL,&success,err) ;
      trueHessian = f->computeCheapHessian(trueHessian,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL;
      }
      if (!success) {
	err = new patErrMiscError("Numerical problem in hessian computation") ;
	WARNING(err->describe()) ;
	return NULL ;
      }

    }
    return trueHessian ;
  }

  // BFGS

  if (mustInitBFGS) {
    mustInitBFGS = patFALSE ;

    if (quasiNewton != NULL) {
      DELETE_PTR(quasiNewton) ;
    }      
    
    if (theParameters.initQuasiNewtonWithTrueHessian && trueHessian != NULL) {
      patHybridMatrix* ptr = trueHessian->getHybridMatrixPtr() ;
      if (ptr == NULL) {
	err = new patErrNullPointer("patHybridMatrix") ;
	WARNING(err->describe()) ;
	return NULL;
      }
      if (theParameters.quasiNewtonUpdate == 2) {
	quasiNewton = new trSR1(*ptr,theParameters,err) ;
	  if (err != NULL) {
	    WARNING(err->describe()) ;
	    return NULL;
	  }
      }
      else {
	quasiNewton = new trBFGS(*ptr,
				 theParameters,
				 err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL;
	}
      }
      return quasiNewton ;
    }
    else {
      if (theParameters.quasiNewtonUpdate == 2) {
	quasiNewton = new trSR1(solution.size(),theParameters,err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL;
	}
      }
      else {
	quasiNewton = new trBFGS(solution.size(),
				 theParameters,
				 err) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return NULL;
	}
      }
    }
    if (quasiNewton == NULL) {
      err = new patErrNullPointer("trSecantUpdate") ;
      WARNING(err->describe()) ;
      return NULL ;
    }
    return quasiNewton ;
  }
  else {
    if (quasiNewton == NULL) {
      err = new patErrNullPointer("trSecantUpdate") ;
      WARNING(err->describe()) ;
      return NULL;
    }
    if (!previousIterate.empty()) {
      trVector sk = currentIterate - previousIterate ;
      quasiNewton->update(sk,
			  currentGradient,
			  previousGradient,
			  *iterInfo,
			  err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return NULL ;
      }
    }
    return quasiNewton ;
  }
}

patBoolean trSimBasedSimpleBoundsAlgo::checkOpt(const trVector& x,
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
  return (gMax < theParameters.tolerance) ;
}

patReal trSimBasedSimpleBoundsAlgo::computeRhoK(patReal fold,
					patReal fnew,
					patReal modelImprovement) {

  patReal rhok  ;
  patReal deltaK = 10.0*patEPSILON*patMax(1.0,patAbs(fold)) ;
  patReal deltaF = fnew - fold - deltaK ;
  patReal deltaM = modelImprovement - deltaK ;
  if (patAbs(deltaF) < 10*patEPSILON 
	&& patAbs(deltaM) < 10.0*patEPSILON) {
    rhok = 1.0 ;
  }
  else {
    rhok = deltaF / deltaM ;
  }

  return rhok ;
}

patVariables trSimBasedSimpleBoundsAlgo::getLowerBoundsLambda() {
  return lowerLambda ;
}

patVariables trSimBasedSimpleBoundsAlgo::getUpperBoundsLambda() {
  return upperLambda ;

}

patString trSimBasedSimpleBoundsAlgo::getName() {
  return patString("Sim. based Trust region algorithm with simple bounds (CGT2000)") ;
}

void trSimBasedSimpleBoundsAlgo::defineStartingPoint(const patVariables& x0) {
  solution = x0 ;
}

patULong trSimBasedSimpleBoundsAlgo::nbrIter() {
  return iter ;
}

