//-*-c++-*------------------------------------------------------------
//
// File name : patSolvOpt.cc
// Author :    \URL[Michel Bierlaire]{http://rosowww.epfl.ch/mbi}
// Date :      Tue Feb  5 11:40:08 2002
//
//--------------------------------------------------------------------

#include <numeric>
#include <map>

#include "patMath.h"
#include "patSolvOpt.h"
#include "solvopt.h"
#include "patNonLinearProblem.h"
#include "patErrNullPointer.h"
#include "trFunction.h"

/**
   @doc global pointer to the problem for the functions used by solvOpt
*/
patNonLinearProblem* globalProblem ;
/**
   @doc global variable identifying the type of constraint with the max residual
*/
patSolvOpt::patMaxConstraintType maxConstraintType ;
/**
   @doc global variable for feasibility
*/
patBoolean isFeasible ;
/**
   @doc global variable for sign of equality constraints. This has an impact on the gradient. Indeed, a constraint $h(x)=0$ is equivalent to $h(x) \leq 0$ and $-h(x)\leq 0$. If the first condition is violated, it is equivalent to inequality constraints. If the second condition is violated, the sign of the gradient must be changed. 
*/
patBoolean changeGradientSign ;

map<patSolvOpt::patMaxConstraintType,patString> constraintTypes ;


/**
   @doc global variable identifying the index of constraint with the max residual. The index is meaningful only for a specific constraint type.
 */
unsigned short maxConstraintIndex ;

/**
 */
patReal solvopt(unsigned short n,
               patReal x[],
               patReal fun(patReal x[]),
               void grad(patReal x[], patReal g[]),
               patReal options[],
               patReal func(patReal x[]),
               void gradc(patReal x[], patReal gc[])
	       ) ;


patSolvOpt::patSolvOpt(solvoptParameters p,
		       patNonLinearProblem* aProblem) :
  trNonLinearAlgo(aProblem),
  theParameters(p),
  startingPoint((aProblem==NULL)?0:aProblem->nVariables()),
  solution((aProblem==NULL)?0:aProblem->nVariables()) {
  globalProblem = aProblem ;

  constraintTypes[patNonLinEq] = "patNonLinEq" ;
  constraintTypes[patLinEq] = "patLinEq" ;
  constraintTypes[patNonLinIneq] = "patNonLinIneq" ;
  constraintTypes[patLinIneq] = "patLinIneq" ;
  constraintTypes[patUpperBound] = "patUpperBound" ;
  constraintTypes[patLowerBound] = "patLowerBound" ;
}
patSolvOpt::~patSolvOpt() {

}

void patSolvOpt::setProblem(patNonLinearProblem* aProblem) {
  theProblem = aProblem ;
  if (theProblem != NULL) {
    solution.resize(theProblem->nVariables()) ;
  }
}

patNonLinearProblem* patSolvOpt::getProblem() {
  return theProblem ;
}

void patSolvOpt::defineStartingPoint(const patVariables& x0) {
  startingPoint=x0 ;
}

patVariables patSolvOpt::getStartingPoint() {
  return startingPoint ;
}
patVariables patSolvOpt::getSolution(patError*& err) {
  return solution ;
}

patReal patSolvOpt::getValueSolution(patError*& err) {
  patBoolean success = patTRUE ;
  if (theProblem == NULL) {
    err = new patErrNullPointer("patNonLinearProblem") ;
    WARNING(err->describe()) ;
    return patReal();

  }
  trFunction* f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (f == NULL) {
    err = new patErrNullPointer("trFunction") ;
    WARNING(err->describe()) ;
    return patReal();
  }

  patReal result = f->computeFunction(&solution,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  if (!success) {
    WARNING("Problem in function evaluation") ;
  }
  return result ;
}

patString patSolvOpt::getName() {
  return patString("SolvOpt") ;
}

patString patSolvOpt::run(patError*& err) {
  if (theProblem == NULL) {
    return patString("No problem defined") ;
  }

  patReal* x = new patReal[theProblem->nVariables()] ;
  copy(startingPoint.begin(),startingPoint.end(),x) ;

  // Set options
  solvopt_options[1] = theParameters.errorArgument ;
  solvopt_options[2] = theParameters.errorFunction ;
  solvopt_options[3] = theParameters.maxIter ;
  solvopt_options[4] = theParameters.display ;

  solvopt(theProblem->nVariables(),
	  x,
	  &ObjectFunctionValue,
	  &ObjectFunctionGradient,
	  solvopt_options,
	  &MaxResidual,
	  &GradMaxResConstr);


  // Number of iterations

  nIter = solvopt_options[8] ;

  // Copy solution

  for (unsigned short i = 0 ; i < theProblem->nVariables() ; ++i) {
    solution[i] = x[i] ;
  }
  delete [] x ;

  theProblem->computeLagrangeMultipliers(solution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }
  
  return patString("Normal termination") ;
}

patReal ObjectFunctionValue(patReal x[]) {

  patError* err = NULL ;
  unsigned short n = globalProblem->nVariables() ;
  patVariables localx(n) ;
  for (unsigned short i = 0 ;
       i < n ;
       ++i) {
    localx[i] = x[i] ;
  }

  //  DEBUG_MESSAGE("x=" << localx) ;

  trFunction* f = globalProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patMaxReal ;
  }
  if (f == NULL) {
    WARNING("Null pointer in obj. fuction evaluation") ;
    return patMaxReal ;    
  }
  patBoolean success ;
  patReal result = f->computeFunction(&localx,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patMaxReal ;
  }
  if (!success) {
    WARNING("Unsuccessful function evaluation") ;
    return patMaxReal ;    
  }
  //  DEBUG_MESSAGE("f(x)=" << result) ;
  return result ;
}

void ObjectFunctionGradient(patReal x[], patReal g[]) {
  patError* err = NULL ;
  unsigned short n = globalProblem->nVariables() ;
  patVariables localx(n) ;
  for (unsigned short i = 0 ;
       i < n ;
       ++i) {
    localx[i] = x[i] ;
  }
  //  DEBUG_MESSAGE("x=" << localx) ;
  trFunction* f = globalProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  if (f == NULL) {
    WARNING("Null pointer in obj. fuction evaluation") ;
    return  ;    
  }
  patBoolean success ;
  trVector result(localx.size());
  f->computeFunctionAndDerivatives(&localx,&result,NULL,&success,err) ;
  //  DEBUG_MESSAGE("g("<< localx<<")=" << result) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return  ;
  }
  if (!success) {
    WARNING("Unsuccessful gradient evaluation") ;
    return  ;    
  }
  copy(result.begin(),result.end(),g) ;
}

patReal MaxResidual(patReal x[]) {
  
  patError* err = NULL ;
  unsigned short n = globalProblem->nVariables() ;
  patVariables localx(n) ;
  for (unsigned short i = 0 ;
       i < n ;
       ++i) {
    localx[i] = x[i] ;
  }
  
  //  DEBUG_MESSAGE("Compute max residual for x=" << localx) ;

  patReal result = 0.0 ;
  isFeasible = patTRUE ;
  changeGradientSign = patFALSE ;

  // Nonlinear inequality
  unsigned short nc = globalProblem->nNonLinearIneq() ;
  for (unsigned short i = 0 ;
       i < nc ;
       ++i) {
    trFunction* f = globalProblem->getNonLinInequality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patNonLinIneq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    if (f == NULL) {
      WARNING("Null pointer in non lin ineq constraint evaluation") ;
      maxConstraintType = patSolvOpt::patNonLinIneq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    patBoolean success ;
    patReal value = f->computeFunction(&localx,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patNonLinIneq ;
      maxConstraintIndex = i ;
      return patMaxReal ;
    }
    if (!success) {
      WARNING("Unsuccessful function evaluation") ;
      maxConstraintType = patSolvOpt::patNonLinIneq ;
      maxConstraintIndex = i ;
      return patMaxReal ;    
    }
    if (value > result) {
      //      DEBUG_MESSAGE("Non Lin Ineq " << i << ": " << value) ;
      maxConstraintType = patSolvOpt::patNonLinIneq ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      changeGradientSign = patFALSE ;
      result = value ;
    }
  }
  // Nonlinear equality
  nc = globalProblem->nNonLinearEq() ;
  for (unsigned short i = 0 ;
       i < nc ;
       ++i) {
    trFunction* f = globalProblem->getNonLinEquality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patNonLinEq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    if (f == NULL) {
      WARNING("Null pointer in obj. fuction evaluation") ;
      maxConstraintType = patSolvOpt::patNonLinEq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    patBoolean success ;
    patReal value = f->computeFunction(&localx,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patNonLinEq ;
      maxConstraintIndex = i ;
      return patMaxReal ;
    }
    if (!success) {
      WARNING("Unsuccessful function evaluation") ;
      maxConstraintType = patSolvOpt::patNonLinEq ;
      maxConstraintIndex = i ;
      return patMaxReal ;    
    }
    if (patAbs(value) > result) {
      //      DEBUG_MESSAGE("Non Lin Eq " << i << ": " << value) ;
      maxConstraintType = patSolvOpt::patNonLinEq ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      result = patAbs(value) ;
      if (value < 0.0) {
	changeGradientSign = patTRUE ;
      }
      else {
	changeGradientSign = patFALSE ;
      }
    }
  }

  //Linear inequality
  nc = globalProblem->nLinearIneq() ;
  for (unsigned short i = 0 ;
       i < nc ;
       ++i) {
    pair<patVariables,patReal> f = globalProblem->getLinInequality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patLinIneq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    patReal value = inner_product(localx.begin(),
				 localx.end(),
				 f.first.begin(),
				 0.0) ;
    value -= f.second ;
    if (value > result) {
      //      DEBUG_MESSAGE("Lin Ineq " << i << ": " << value) ;
      maxConstraintType = patSolvOpt::patLinIneq ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      changeGradientSign = patFALSE ;
      result = value ;
    }
  }

  //Linear equality
  nc = globalProblem->nLinearEq() ;
  for (unsigned short i = 0 ;
       i < nc ;
       ++i) {
    pair<patVariables,patReal> f = globalProblem->getLinEquality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      maxConstraintType = patSolvOpt::patLinEq ;
      maxConstraintIndex = i ;
      return patMaxReal;
    }
    patReal value = inner_product(localx.begin(),
				 localx.end(),
				 f.first.begin(),
				 0.0) ;
    value -= f.second ;
    if (patAbs(value) > result) {
      //            DEBUG_MESSAGE("Lin Eq " << i << ": " << value) ;
      maxConstraintType = patSolvOpt::patLinEq ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      result = patAbs(value) ;
      if (value < 0.0) {
	changeGradientSign = patTRUE ;
      }
      else {
	changeGradientSign = patFALSE ;
      }
    }
  }

  // Bounds 
  
  static patVariables upperBounds = globalProblem->getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    maxConstraintType = patSolvOpt::patUpperBound ;
    maxConstraintIndex = 0 ;
    return patMaxReal;
  }
  static patVariables lowerBounds = globalProblem->getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    maxConstraintType = patSolvOpt::patLowerBound ;
    maxConstraintIndex = 0 ;
    return patMaxReal;
  }

  for (unsigned short i = 0 ;
       i < n ;
       ++i) {
    patReal ub = x[i] - upperBounds[i] ;
    if (ub > result) {
      maxConstraintType = patSolvOpt::patUpperBound ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      changeGradientSign = patFALSE ;
      result = ub ;
    }
    patReal lb = - x[i] + lowerBounds[i] ;
    if (lb > result) {
      maxConstraintType = patSolvOpt::patLowerBound ;
      maxConstraintIndex = i ;
      isFeasible = patFALSE ;
      changeGradientSign = patFALSE ;
      result = lb ;
    }
  }

  return result ;
}

void GradMaxResConstr(patReal x[], patReal gc[]) {

  //  DEBUG_MESSAGE("Computes the gradient of the max residual constraint") ;

  // WARNING: the implementation of this function assumes that the function
  // MaxResidual has been previously called with the same vector x, and that
  // the variables maxConstraintType and maxConstraintIndex are correctly
  // set. This assumption comes from the nlpsmple.c example in the SolvOpt
  // distribution.


  patError* err = NULL ;
  unsigned short n = globalProblem->nVariables() ;
  if (isFeasible) {
    for (unsigned short i = 0 ;
	 i < n ;
	 ++i) {
      gc[i] = 0.0 ;
    }
    return ;
  }
  patVariables localx(n) ;
  for (unsigned short i = 0 ;
       i < n ;
       ++i) {
    localx[i] = x[i] ;
  }

  switch (maxConstraintType) {
  case patSolvOpt::patNonLinEq:
    {
      trFunction* f = globalProblem->getNonLinEquality(maxConstraintIndex,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (f == NULL) {
	WARNING("Null pointer") ;
	return  ;    
      }
      patBoolean success ;
      trVector result(localx.size());
      f->computeFunctionAndDerivatives(&localx,&result,NULL,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      if (!success) {
	WARNING("Unsuccessful gradient evaluation") ;
	return  ;    
      }
      copy(result.begin(),result.end(),gc) ;
      //      DEBUG_MESSAGE("patNonLinEq[" << maxConstraintIndex <<"]") ;
      //      DEBUG_MESSAGE("grad=" << result) ;
      if (changeGradientSign) {
	for (unsigned short i = 0 ;
	     i < n ;
	     ++i) {
	  gc[i] = -gc[i] ;
	}
      }
      return ;
    }
  case patSolvOpt::patLinEq:
    {
      pair<patVariables,patReal> result = 
	globalProblem->getLinEquality(maxConstraintIndex,
				     err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      copy(result.first.begin(),result.first.end(),gc) ;
      if (changeGradientSign) {
	for (unsigned short i = 0 ;
	     i < n ;
	     ++i) {
	  gc[i] = -gc[i] ;
	}
      }
      return ;
    }
  case patSolvOpt::patNonLinIneq:
    {
      trFunction* f = globalProblem->getNonLinInequality(maxConstraintIndex,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (f == NULL) {
	WARNING("Null pointer") ;
	return  ;    
      }
      patBoolean success ;
      trVector result(localx.size());
      f->computeFunctionAndDerivatives(&localx,&result,NULL,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      if (!success) {
	WARNING("Unsuccessful gradient evaluation") ;
	return  ;    
      }
      copy(result.begin(),result.end(),gc) ;
      //      DEBUG_MESSAGE("patNonLinIneq[" << maxConstraintIndex << "]") ;
      //      DEBUG_MESSAGE("grad=" << result) ;
      if (changeGradientSign) {
	WARNING("Gradient should not be changed for inequalities. Check...") ;
      }
      return ;
    }
  case patSolvOpt::patLinIneq:
    {
      pair<patVariables,patReal> result = 
	globalProblem->getLinInequality(maxConstraintIndex,
				     err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return  ;
      }
      copy(result.first.begin(),result.first.end(),gc) ;
      //      DEBUG_MESSAGE("patLinIneq[" << maxConstraintIndex << "]") ;
      //      DEBUG_MESSAGE("grad=" << result.first) ;
      if (changeGradientSign) {
	WARNING("Gradient should not be changed for inequalities. Check...") ;
      }
      return ;
    }
  case patSolvOpt::patUpperBound:
    {

      patVariables debugOnly(n,0.0) ;
      for (unsigned short i = 0 ;
	   i < n ;
	   ++i) {
	gc[i] = 0.0 ;
      }
      gc[maxConstraintIndex] = 1.0 ;
      debugOnly[maxConstraintIndex] = 1.0 ;
      //      DEBUG_MESSAGE("Upper bound[" << maxConstraintIndex <<"]") ;
      //      DEBUG_MESSAGE("g(x)=" << debugOnly) ;
      if (changeGradientSign) {
	WARNING("Gradient should not be changed for inequalities. Check...") ;
      }
      return ;
    }
  case patSolvOpt::patLowerBound:
      patVariables debugOnly(n,0.0) ;
      for (unsigned short i = 0 ;
	   i < n ;
	   ++i) {
	gc[i] = 0.0 ;
      }

      gc[maxConstraintIndex] = -1.0 ;
      debugOnly[maxConstraintIndex] = -1.0 ;
      if (changeGradientSign) {
	WARNING("Gradient should not be changed for inequalities. Check...") ;
      }
      return ;
  };

}

patVariables patSolvOpt::getLowerBoundsLambda() {
  WARNING("Dual variables unavailable") ;
  return patVariables() ;
}
patVariables patSolvOpt::getUpperBoundsLambda() {
  WARNING("Dual variables unavailable") ;
  return patVariables() ;
}

patULong patSolvOpt::nbrIter() {
  return nIter ;
}
