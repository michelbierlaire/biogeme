//-*-c++-*------------------------------------------------------------
//
// File name : bioIpopt.cc
// Author :    Michel Bierlaire
// Date :      Wed Mar 22 06:47:05 2017
//
//--------------------------------------------------------------------

#include "bioIpopt.h"
#include "patDisplay.h"
#include "patNonLinearProblem.h"
#include "patErrMiscError.h"
#include "trVector.h"
#include "trFunction.h"
#include "bioParameters.h"
#include "patIterationBackup.h"

#ifdef IPOPT
#include "IpSolveStatistics.hpp"
#endif

bioIpopt::bioIpopt(patIterationBackup* i, patNonLinearProblem* aProblem):
  trNonLinearAlgo(aProblem) ,
  startingPointDefined(patFALSE),
  theInteraction(i) {

  
#ifdef IPOPT
  app = IpoptApplicationFactory();
#endif

  if (theProblem != NULL) {
    n = theProblem->nVariables() ;
  }
  else {
    n = 0 ;
    return ;
  }
  patError* err(NULL) ;
  exactHessian =  (bioParameters::the()->getValueInt("useAnalyticalHessianForOptimization",err)) != 0 ;
  if (exactHessian) {
    theHessian = new trHessian(bioParameters::the()->getTrParameters(err),n) ;
  }
  else {
    theHessian = NULL ;
  }

}

bioIpopt::~bioIpopt() {

}

#ifdef IPOPT
// Methods required by IPOPT
  
bool bioIpopt::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
			    Index& nnz_h_lag, IndexStyleEnum& index_style) {

  if (theProblem == NULL) {
    return false ;
  }
  n = theProblem->nVariables() ;
  m = theProblem->nNonTrivialConstraints() ;
  if (m > 0) {
    WARNING("IPOPT interface for general constraints not yet implemented") ;
    return false ;
  }
  nnz_jac_g = n * m ;
  nnz_h_lag = n * (n+1) / 2 ;
  index_style = TNLP::C_STYLE;
  return true ;
}
  

bool bioIpopt::get_bounds_info(Index n, Number* x_l, Number* x_u,
			       Index m, Number* g_l, Number* g_u) {

  if (theProblem == NULL) {
    return false ;
  }
  patError* err(NULL) ;
  patVariables lb = theProblem->getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  if (m > 0) {
    err = new patErrMiscError("IPOPT interface for general constraints not yet implemented") ;
    WARNING(err->describe()) ;
    return false ;
  }
  if (lb.size() != n) {
    stringstream str ;
    str << "Incompatible sizes: " << n << " and " << lb.size() << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return false ;
  }
  for (Index i=0; i<n; ++i) {
    x_l[i] = lb[i];
  }

  patVariables ub = theProblem->getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  if (ub.size() != n) {
    stringstream str ;
    str << "Incompatible sizes: " << n << " and " << ub.size() << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return false ;
  }
  for (Index i=0; i<n; ++i) {
    x_u[i] = ub[i];
  }

  return true ;
}

bool bioIpopt::get_starting_point(Index n, bool init_x, Number* x,
				  bool init_z, Number* z_L, Number* z_U,
				  Index m, bool init_lambda,
				  Number* lambda) {

  patError* err(NULL) ;
  if (!init_x) {
    return false;
  }
  if (init_z) {
    err = new patErrMiscError("IPOPT cannot be configured to require init values for the bounds mulitpliers") ;
    WARNING(err->describe()) ;
    return false ;
  }
  if (init_lambda) {
    err = new patErrMiscError("IPOPT cannot be configured to require init values for the dual variables") ;
    WARNING(err->describe()) ;
    return false ;
  }
  if (!startingPointDefined) {
    err = new patErrMiscError("No starting point has been defined") ;
    WARNING(err->describe()) ;
    return false ;
  }
  if (startingPoint.size() != n) {
    stringstream str ;
    str << "Incompatible sizes: " << n << " and " << startingPoint.size() << endl ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return false ;
  }
  for (Index i=0; i<n; ++i) {
    x[i] = startingPoint[i];
  }
  return true ;
}
  
  /** Method to return the objective value */
bool bioIpopt::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
  patError* err(NULL) ;
  trFunction* f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  patBoolean success(patFALSE) ;
  trVector xstl(n) ;
  for (unsigned short i = 0 ; i < n ; ++i) {
    xstl[i] = x[i] ;
  }
  obj_value = f->computeFunction(&xstl,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  return success ;
}
  
  /** Method to return the gradient of the objective */
bool bioIpopt::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f){
  theInteraction->saveCurrentIteration() ;
  patError* err(NULL) ;
  trFunction* f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  patBoolean success(patFALSE) ;
  trVector xstl(n) ;
  for (unsigned short i = 0 ; i < n ; ++i) {
    xstl[i] = x[i] ;
  }
  trVector grad(n) ;
  f->computeFunctionAndDerivatives(&xstl,&grad,NULL,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return false ;
  }
  for (unsigned short i = 0 ; i < n ; ++i) {
    grad_f[i] = grad[i] ;
  }
  return success ;
  
}
  
  /** Method to return the constraint residuals */
bool bioIpopt::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
  // No general constraints implemented for now.
  return false ;
}
  
  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
bool bioIpopt::eval_jac_g(Index n, const Number* x, bool new_x,
			  Index m, Index nele_jac, Index* iRow, Index *jCol,
			  Number* values) {
  // No general constraints implemented for now.
  return false ;
}
  
  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
bool bioIpopt::eval_h(Index n, const Number* x, bool new_x,
		      Number obj_factor, Index m, const Number* lambda,
		      bool new_lambda, Index nele_hess, Index* iRow,
		      Index* jCol, Number* values) {


  // Let's try without the hessian first. 
  return false ;
}
  
  //@}
  
  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
void bioIpopt::finalize_solution(SolverReturn status,
				 Index n, const Number* x, const Number* z_L, const Number* z_U,
				 Index m, const Number* g, const Number* lambda,
				 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq) {

  solutionValue = obj_value ;
  solution.resize(n) ;
  solutionGradient.resize(n) ;
  lowerLambda.resize(n) ;
  upperLambda.resize(n) ;
  for (unsigned short i = 0 ; i < n ; ++i) {
    solution[i] = x[i] ;
    solutionGradient[i] = g[i] ;
    lowerLambda[i] = z_L[i] ;
    upperLambda[i] = z_U[i] ;
  }
  finalStatus = status ;

}
#endif
  // Methods required by Biogeme
  /**
     @return Diagnostic from algorithm
   */
patString bioIpopt::run(patError*& err)  {

#ifdef IPOPT
 app->Options()->SetStringValue("hessian_approximation", "limited-memory");
 app->Options()->SetStringValue("output_file", "ipopt.out");
 app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
 app->Options()->SetStringValue("nlp_scaling_method", "gradient-based");

 // User defined options
 patReal IPOPTtol = bioParameters::the()->getValueReal("IPOPTtol",err) ;
 app->Options()->SetNumericValue("tol", IPOPTtol);
 patULong IPOPTmax_iter = bioParameters::the()->getValueInt("IPOPTmax_iter",err) ;
 app->Options()->SetIntegerValue("max_iter", IPOPTmax_iter);
 patReal IPOPTmax_cpu_time = bioParameters::the()->getValueReal("IPOPTmax_cpu_time",err) ;
 app->Options()->SetNumericValue("max_cpu_time", IPOPTmax_cpu_time);
 patReal IPOPTacceptable_tol = bioParameters::the()->getValueReal("IPOPTacceptable_tol",err) ;
 app->Options()->SetNumericValue("acceptable_tol", IPOPTacceptable_tol);
 patULong IPOPTacceptable_iter = bioParameters::the()->getValueInt("IPOPTacceptable_iter",err) ;
 app->Options()->SetIntegerValue("acceptable_iter", IPOPTacceptable_iter);

// Initialize the IpoptApplication and process the options
 ApplicationReturnStatus status = app->Initialize();
	
 // Ask Ipopt to solve the problem
 status = app->OptimizeTNLP(this);
 return getFinalStatus() ;
#else
 return("Biogeme has not been compiled with IPOPT") ;
#endif
}
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */
patULong bioIpopt::nbrIter() {
#ifdef IPOPT
  SmartPtr<SolveStatistics> stats = app->Statistics();
  return(stats->IterationCount()) ;
#else
  return 0 ;
#endif
  
}

  
patVariables bioIpopt::getSolution(patError*& err) {
  DEBUG_MESSAGE("Solution of size: " << solution.size()) ;
  DEBUG_MESSAGE("Solution: " << solution) ;
  return solution ;
}
patReal bioIpopt::getValueSolution(patError*& err) {
  return solutionValue ;
}

patVariables bioIpopt::getLowerBoundsLambda() {
  return lowerLambda ;
}
patVariables bioIpopt::getUpperBoundsLambda() {
  return upperLambda ;
}

void bioIpopt::defineStartingPoint(const patVariables& x0)  {
  startingPoint = x0 ;
  startingPointDefined = patTRUE ;
}

patString bioIpopt::getName() {
  return patString("IPOPT") ;
}


#ifdef IPOPT
patString bioIpopt::getFinalStatus() {
  switch (finalStatus) {
  case SUCCESS:
    return patString("Optimal Solution Found") ;
  case MAXITER_EXCEEDED:
    return patString("Maximum Number of Iterations Exceeded") ;
  case CPUTIME_EXCEEDED:
    return patString("Maximum CPU time exceeded") ;
  case STOP_AT_TINY_STEP:
    return patString("Search Direction is becoming Too Small") ;
  case STOP_AT_ACCEPTABLE_POINT:
    return patString("Solved To Acceptable Level") ;
  case LOCAL_INFEASIBILITY:
    return patString("Converged to a point of local infeasibility. Problem may be infeasible.") ;
  case USER_REQUESTED_STOP:
    return patString("Stopping optimization at current point as requested by user.") ;
  case FEASIBLE_POINT_FOUND:
    return patString("Feasible point for square problem found.") ;
  case DIVERGING_ITERATES:
    return patString("Iterates divering; problem might be unbounded. ") ;
  case RESTORATION_FAILURE:
    return patString("Restoration Failed!") ;
  case ERROR_IN_STEP_COMPUTATION:
    return patString("Error in step computation (regularization becomes too large?)! ") ;
  case INVALID_NUMBER_DETECTED:
    return patString("Invalid number detected.") ;
  case TOO_FEW_DEGREES_OF_FREEDOM:
    return patString("Problem has too few degrees of freedom.") ;
  case INVALID_OPTION:
    return patString("Invalid option") ;
  case OUT_OF_MEMORY:
    return patString("Not enough memory.") ;
  case INTERNAL_ERROR:
    return patString("INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors.") ;
  case UNASSIGNED:
    return patString("Unknown return status") ;
    
  }
}

#endif

patBoolean bioIpopt::isAvailable() const {
#ifdef IPOPT
  return patTRUE ;
#else
  return patFALSE ;
#endif
}
