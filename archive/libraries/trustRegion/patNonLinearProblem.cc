//-*-c++-*------------------------------------------------------------
//
// File name : patNonLinearProblem.cc
// Author :    Michel Bierlaire
// Date :      Mon Apr 30 14:03:33 2001
//
//--------------------------------------------------------------------

#include <sstream>
#include <numeric>

#include "patMyMatrix.h"
#include "patMath.h"
#include "patLu.h"
#include "trFunction.h"
#include "patNonLinearProblem.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"

unsigned long patNonLinearProblem::nConstraints() {
  return(2*nVariables()   +  // lower and upper bounds
	 nNonLinearEq()   +
	 nNonLinearIneq() +
	 nLinearEq()      +
	 nLinearIneq()    ) ;
}

unsigned long patNonLinearProblem::nNonTrivialConstraints() {
  return(nNonLinearEq()   +
	 nNonLinearIneq() +
	 nLinearEq()      +
	 nLinearIneq()    ) ;
}

void patNonLinearProblem::setLagrangeLowerBounds(const patVariables& l,
						 patError*&  err) {
  if (l.size() != nVariables()) {
    stringstream str ;
    str << "Nbr of variables is " << nVariables() 
	<< " and not " << l.size()  ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeLowerBounds = l ;
}

void patNonLinearProblem::setLagrangeUpperBounds(const patVariables& l,
						 patError*&  err) {
  if (l.size() != nVariables()) {
    stringstream str ;
    str << "Nbr of variables is " << nVariables() 
	<< " and not " << l.size()  ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeUpperBounds = l ;
}

void patNonLinearProblem::setLagrangeNonLinEq(const patVariables& l,
					      patError*&  err) {
  if (l.size() != nNonLinearEq()) {
    stringstream str ;
    str << "Nbr of non-linear equality constraints is " << nNonLinearEq() 
	<< " and not " << l.size()  ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeNonLinEqConstraints = l ;
}

void patNonLinearProblem::setLagrangeLinEq(const patVariables& l,
					   patError*&  err) {
  if (l.size() != nLinearEq()) {
    stringstream str ;
    str << "Nbr of linear equality constraints is " << nLinearEq() 
	<< " and not " << l.size()  ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeLinEqConstraints = l ;
}

void patNonLinearProblem::setLagrangeNonLinIneq(const patVariables& l,
					   patError*&  err) {
  if (l.size() != nNonLinearIneq()) {
    stringstream str ;
    str << "Nbr of non-linear inequality constraints is " << nNonLinearIneq() 
	<< " and not " << l.size() ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeNonLinIneqConstraints = l ;
}

void patNonLinearProblem::setLagrangeLinIneq(const patVariables& l,
					   patError*&  err) {
  if (l.size() != nLinearIneq()) {
    stringstream str ;
    str << "Nbr of linear inequality constraints is " << nLinearIneq() 
	<< " and not " << l.size()  ;
    err = new patErrMiscError(str.str()) ;
    return ;
  }
  lagrangeLinIneqConstraints = l ;
}

patVariables patNonLinearProblem::getLagrangeLowerBounds() {
  return lagrangeLowerBounds ;
}

patVariables patNonLinearProblem::getLagrangeUpperBounds() {
  return lagrangeUpperBounds ;
}

patVariables patNonLinearProblem::getLagrangeNonLinEq() {
  return lagrangeNonLinEqConstraints ;
}

patVariables patNonLinearProblem::getLagrangeLinEq() {
  return lagrangeLinEqConstraints ;
}

patVariables patNonLinearProblem::getLagrangeNonLinIneq() {
  return lagrangeNonLinIneqConstraints ;
}

patVariables patNonLinearProblem::getLagrangeLinIneq() {
  return lagrangeLinIneqConstraints ;
}



patString patNonLinearProblem::getVariableName(unsigned long i,patError*&  err) {
  stringstream str ;
  str << "x" << i ;
  return patString(str.str()) ;
}

patString patNonLinearProblem::getNonLinIneqConstrName(unsigned long i,patError*&  err) {
  stringstream str ;
  str << "nli" << i ;
  return patString(str.str()) ;
}
patString patNonLinearProblem::getNonLinEqConstrName(unsigned long i,patError*&  err) {
  stringstream str ;
  str << "nle" << i ;
  return patString(str.str()) ;
}

patString patNonLinearProblem::getLinIneqConstrName(unsigned long i, patError*&  err) {
  stringstream str ;
  str << "li" << i ;
  return patString(str.str()) ;
}

patString patNonLinearProblem::getLinEqConstrName(unsigned long i,patError*&  err) {
  stringstream str ;
  str << "le" << i  ;
  return patString(str.str()) ;
}

void patNonLinearProblem::computeLagrangeMultipliers(patVariables& xOpt,
						     patError*&  err) {

  // First, identify active constraints 
  // We check also feasibility...

  patReal activityThreshold = patSQRT_EPSILON ;

  vector<unsigned long> activeNonLinear ;
  vector<patVariables> gradientNonLinear ;
  vector<unsigned long> activeLinear ;
  vector<unsigned long> activeLower ;
  vector<unsigned long> activeUpper ;

  unsigned long totalActive = 0 ;
  patBoolean success ;

  // Nonlinear equality   h(x) = 0 
  
  if (nNonLinearEq() != 0) {
    lagrangeNonLinEqConstraints.resize(nNonLinearEq(),0.0) ;
    for (unsigned long i = 0 ; 
	 i < nNonLinearEq() ;
	 ++i) {
      trFunction* h = getNonLinEquality(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal hval = h->computeFunction(&xOpt,
					&success,
					err) ;
      if (!success) {
	stringstream str ;
	str << "Unable to compute nonlinear equality constraint " 
	    << getNonLinEqConstrName(i,err) ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (patAbs(hval) > activityThreshold) {
	stringstream str ;
	str << "Nonlinear equality constraint " 
		<< getNonLinEqConstrName(i,err) 
	    << " not verified" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      else {
	++totalActive ;
      }
    }
  }

  // Nonlinear inequality g(x) <= 0
  

  if (nNonLinearIneq() != 0) {
    lagrangeNonLinIneqConstraints.resize(nNonLinearIneq(),0.0) ;
    for (unsigned long i = 0 ; 
	 i < nNonLinearIneq() ;
	 ++i) {
      trFunction* g = getNonLinInequality(i,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal gval = g->computeFunction(&xOpt,
					&success,
					err) ;
      if (!success) {
	stringstream str ;
	str << "Unable to compute nonlinear inequality constraint " 
	    << getNonLinIneqConstrName(i,err) ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      if (gval > activityThreshold) {
	stringstream str ;
	str << "Nonlinear inequality constraint " 
	    << getNonLinIneqConstrName(i,err) 
	    << " not verified" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      else if (gval > -activityThreshold) {
	++totalActive ;
	activeNonLinear.push_back(i) ;
      }
    }
  }

  // Linear equalities Ax - b = 0

  if (nLinearEq() != 0) {
    lagrangeLinEqConstraints.resize(nLinearEq(),0.0) ;
    for (unsigned long i = 0 ;
	 i < nLinearEq() ;
	 ++i) {
      pair<patVariables,patReal> constraint = getLinEquality(i,err); 
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal axmb = inner_product(constraint.first.begin(),
				   constraint.first.end(),
				   xOpt.begin(),
				   0.0) - constraint.second ;
      if (patAbs(axmb) > activityThreshold) {
	stringstream str ;
	str << "Linear equality constraint " 
	    << getLinEqConstrName(i,err) 
	    << " not verified" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      else {
	++totalActive ;
      }
    }
  }

  // Linear inequalities Cx-d <= 0

  if (nLinearIneq() != 0) {
    lagrangeLinIneqConstraints.resize(nLinearIneq(),0.0) ;
    for (unsigned long i = 0 ;
	 i < nLinearIneq() ;
	 ++i) {
      pair<patVariables,patReal> constraint = getLinInequality(i,err); 
      if (err != NULL) {
	WARNING(err->describe()) ;
	return ;
      }
      patReal axmb = inner_product(constraint.first.begin(),
				   constraint.first.end(),
				   xOpt.begin(),
				   0.0) - constraint.second ;
      if (axmb > activityThreshold) {
	stringstream str ;
	str << "Linear inequality constraint " 
	    << getLinIneqConstrName(i,err) 
	    << " not verified" ;
	err = new patErrMiscError(str.str()) ;
	WARNING(err->describe());
	return ;
      }
      else if (axmb > -activityThreshold) {
	++totalActive ;
	activeLinear.push_back(i) ;
      }
    }
  }

  // Bounds

  patVariables lower = getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patVariables upper = getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  lagrangeLowerBounds.resize(nVariables(),0.0) ;
  lagrangeUpperBounds.resize(nVariables(),0.0) ;
  for (unsigned long i = 0 ;
       i < nVariables() ;
       ++i) {
    // We check the lower bound first for activity. If, by any chance, the
    // lower bound and the upper bound are both active, we consider only the
    // lower one.

    if (patAbs(lower[i]-xOpt[i]) < activityThreshold) {
      ++totalActive ;
      activeLower.push_back(i) ;
    }
    else if (patAbs(xOpt[i]-upper[i]) < activityThreshold) {
      ++totalActive ;
      activeUpper.push_back(i) ;
    }
    else if (xOpt[i] < lower[i]) {
      stringstream str ;
      str << "Lower bound " << i  
	  << " not verified"  ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
    else if (xOpt[i] > upper[i]) {
      stringstream str ;
      str << "Upper bound " << i  
	  << " not verified" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
  }

  DEBUG_MESSAGE("There are " << totalActive << " active constraints") ;

  if (totalActive == 0) {
    return ;
  }

  patMyMatrix gradConstraint(nVariables(),totalActive) ;

  unsigned long currentColumn = 0;

  // Nonlinear equality   h(x) = 0 
  
  for (unsigned long i = 0 ; 
       i < nNonLinearEq() ;
       ++i) {
    trFunction* h = getNonLinEquality(i,err) ;
    patVariables g(h->getDimension()) ;
    h->computeFunctionAndDerivatives(&xOpt,&g,NULL,&success,err) ;
    if (!success) {
      stringstream str ;
      str << "Unable to compute nonlinear equality constraint " 
	  << getNonLinEqConstrName(i,err) ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = g[j] ;
    }
    ++currentColumn ;
  }

  // Active nonlinear inequalities   g(x) = 0 

  for (vector<unsigned long>::iterator i = activeNonLinear.begin() ;
       i != activeNonLinear.end() ;
       ++i) {
    trFunction* h = getNonLinInequality(*i,err) ;
    patVariables g(h->getDimension()) ; 
    h->computeFunctionAndDerivatives(&xOpt,&g,NULL,&success,err) ;
    if (!success) {
      stringstream str ;
      str << "Unable to compute nonlinear inequality constraint " 
	  << getNonLinIneqConstrName(*i,err) ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe());
      return ;
    }
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = g[j] ;
    }
    ++currentColumn ;
  }

  // Linear equality Ax - b = 0

  for (unsigned long i = 0 ;
       i < nLinearEq() ;
       ++i) {
    pair<patVariables,patReal> constraint = getLinEquality(i,err); 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = constraint.first[j] ;
    }
    ++currentColumn ;
    
  }
  // Active linear equalities Cx - d = 0

  for (vector<unsigned long>::iterator i = activeLinear.begin() ;
       i != activeLinear.end() ;
       ++i) {

    pair<patVariables,patReal> constraint = getLinInequality(*i,err); 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = constraint.first[j] ;
    }
    ++currentColumn ;
  }

  // Active lower bounds

  for (vector<unsigned long>::iterator i = activeLower.begin() ;
       i != activeLower.end() ;
       ++i) {
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = 0.0 ;
    }
    gradConstraint[*i][currentColumn] = -1.0 ;
    ++currentColumn ;
  }

  // Active upper bounds

  for (vector<unsigned long>::iterator i = activeUpper.begin() ;
       i != activeUpper.end() ;
       ++i) {
    for (unsigned long j = 0 ;
	 j < nVariables() ;
	 ++j) {
      gradConstraint[j][currentColumn] = 0.0 ;
    }
    gradConstraint[*i][currentColumn] = 1.0 ;
    ++currentColumn ;
  }

  // Gradient of objective function

  trFunction* f = getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  patVariables grad(f->getDimension()) ;
  f->computeFunctionAndDerivatives(&xOpt,&grad,NULL,&success,err) ;
  if (!success) {
    stringstream str ;
    str << "Unable to compute objective function's gradient " ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe());
    return ;
  }
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  
  patVariables b(totalActive) ;
  patVariables lambda(totalActive) ;
  patVariables theGradient(grad.size()) ;
  for (unsigned long i = 0 ; i < grad.size() ; ++i) {
    theGradient[i] = grad[i] ;
  }

  multTranspVec(gradConstraint,theGradient,b,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patMyMatrix cTc(totalActive,totalActive) ;
  multATranspB(gradConstraint,gradConstraint,cTc,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }

  patLu lu(&cTc) ;
  lambda= *lu.solve(b,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  for (unsigned long i = 0 ;
       i < lambda.size() ;
       ++i) {
    if (!patFinite(lambda[i])) {
      WARNING("The constraint qualification condition is not verified. Unable to compute dual variables from KKT conditions") ;
      return ;
    }
  }
  

  // At this point, we have the dual variables (if all assumptions where verified)
  // We now put them in thew appropriate place
  
  // Nonlinear equality   h(x) = 0 
  
  currentColumn = 0 ;

  for (unsigned long i = 0 ; 
       i < nNonLinearEq() ;
       ++i) {
    lagrangeNonLinEqConstraints[i] = lambda[currentColumn] ;
    ++currentColumn ;
  }  
  
  // Active nonlinear inequalities   g(x) = 0 

  for (vector<unsigned long>::iterator i = activeNonLinear.begin() ;
       i != activeNonLinear.end() ;
       ++i) {
    lagrangeNonLinIneqConstraints[*i] = lambda[currentColumn] ;
    ++currentColumn ;
  }

  // Linear equality Ax - b = 0

  for (unsigned long i = 0 ;
       i < nLinearEq() ;
       ++i) {
    lagrangeLinEqConstraints[i] = lambda[currentColumn] ;
    ++currentColumn ;
  }

  // Active linear equalities Cx - d = 0

  for (vector<unsigned long>::iterator i = activeLinear.begin() ;
       i != activeLinear.end() ;
       ++i) {
    lagrangeLinIneqConstraints[*i] = lambda[currentColumn] ;
    ++currentColumn ;
  }

  // Active lower bounds

  for (vector<unsigned long>::iterator i = activeLower.begin() ;
       i != activeLower.end() ;
       ++i) {
    lagrangeLowerBounds[*i] = lambda[currentColumn] ;
    ++currentColumn ;
  }

  // Active upper bounds

  for (vector<unsigned long>::iterator i = activeUpper.begin() ;
       i != activeUpper.end() ;
       ++i) {
    lagrangeUpperBounds[*i] = lambda[currentColumn] ;
    ++currentColumn ;
  }

}



unsigned long  patNonLinearProblem::getNumberOfActiveConstraints(patVariables& x,
								patError*& err) {
  if (x.size() != nVariables()) {
    stringstream str ;
    str << "Incompatible sizes: " << x.size() << " and " 
	<< nVariables() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return 9999 ;
  }

  unsigned long n(0) ;
  
  // Equality constraints

  n += nNonLinearEq() ;
  n += nLinearEq() ;
  
  // Bound constraints

  patVariables l = getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return 9999 ;
  }
  patVariables u = getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return 9999 ;
  }

  for (unsigned i = 0 ; i < x.size() ; ++i) {
    if (patAbs(x[i] - l[i]) <= patEPSILON) {
      ++n ;
    }
    if (patAbs(u[i]-x[i]) <= patEPSILON) {
      ++n ;
    }
  }

  // Linear inequalities

  for (unsigned i = 0 ; i < nLinearIneq() ; ++i) {
    pair<patVariables,patReal> c = 
      getLinInequality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return 9999 ;
    }

    patReal ip = inner_product(x.begin(),x.end(),c.first.begin(),0.0) ;
    if (patAbs(ip - c.second) <= patEPSILON) {
      ++n ;
    }
  }

  // Non linear inequalities

  for (unsigned long i = 0 ; i < nNonLinearIneq() ; ++i) {

    trFunction* h = getNonLinInequality(i,err) ; 
    if (err != NULL) {
      WARNING(err->describe()) ;
      return 9999 ;
    }
    if (h == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return 9999 ;
    }
    patBoolean success ;
    patReal hval = h->computeFunction(&x,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return 9999 ;
    }
    if (!success) {
      stringstream str ;
      str << "Computation of nonlinear inequality constraint " << i 
	  << " was unsuccessful" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe())  ;
      return 9999 ;
    }
    if (patAbs(hval) <= patEPSILON) {
      ++n ;
    }
  }
  return n ;
}
