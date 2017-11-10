//-*-c++-*------------------------------------------------------------
//
// File name : patCfsqp.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr 27 07:42:20 2001
//
//--------------------------------------------------------------------

//#define FINDIFF

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <sstream>
#include <numeric>
#include "trFunction.h"
#include "patMath.h"
#include "patErrMiscError.h"
#include "patErrNullPointer.h"
#include "patCfsqp.h"
#include "patNonLinearProblem.h"
#include "cfsqpusr.h"
#include "patIterationBackup.h"

patCfsqp::patCfsqp(patIterationBackup* i, patNonLinearProblem* aProblem) :
  trNonLinearAlgo(aProblem),
  startingPoint((aProblem==NULL)?0:aProblem->nVariables()),
  solution((aProblem==NULL)?0:aProblem->nVariables()),
  lowerBoundsLambda((aProblem==NULL)?0:aProblem->nVariables()),
  upperBoundsLambda((aProblem==NULL)?0:aProblem->nVariables()),
  nonLinIneqLambda((aProblem==NULL)?0:aProblem->nNonLinearIneq()),
  linIneqLambda((aProblem==NULL)?0:aProblem->nLinearIneq()),
  nonLinEqLambda((aProblem==NULL)?0:aProblem->nNonLinearEq()),
  linEqLambda((aProblem==NULL)?0:aProblem->nLinearEq()),
  mode(100),
  iprint(2),
  miter(1000),
  eps(6.05545e-06),
  epseqn(6.05545e-06),
  udelta(0.0) ,
  theInteraction(i),
  stopFile("STOP"),
  nIter(static_cast<int>(patBadId)) 
{

}
 patCfsqp::~patCfsqp() {

}

void patCfsqp::setProblem(patNonLinearProblem* aProblem) {
  theProblem = aProblem ;
  if (theProblem != NULL) {
    solution.resize(theProblem->nVariables()) ;
    lowerBoundsLambda.resize(theProblem->nVariables()) ;
    upperBoundsLambda.resize(theProblem->nVariables()) ;
    nonLinIneqLambda.resize(theProblem->nNonLinearIneq()) ;
    linIneqLambda.resize(theProblem->nLinearIneq()) ;
    nonLinEqLambda.resize(theProblem->nNonLinearEq()) ;
    linEqLambda.resize(theProblem->nLinearEq()) ;
  }
}

patNonLinearProblem* patCfsqp::getProblem() {
  return theProblem ;
}

void patCfsqp::defineStartingPoint(const patVariables& x0) {
  startingPoint=x0 ;
}

patVariables patCfsqp::getStartingPoint() {
  return startingPoint ;
}

patVariables patCfsqp::getSolution(patError*& err) {
  return solution ;
}

patVariables patCfsqp::getLowerBoundsLambda() {
  return lowerBoundsLambda ;
}
patVariables patCfsqp::getUpperBoundsLambda() {
  return upperBoundsLambda ;
}
patVariables patCfsqp::getNonLinIneqLambda() {
  return nonLinIneqLambda ;
}

patVariables patCfsqp::getLinIneqLambda() {
  return linIneqLambda ;
}
patVariables patCfsqp::getNonLinEqLambda() {
  return nonLinEqLambda ;
}
patVariables patCfsqp::getLinEqLambda() {
  return linEqLambda ;
}



patString patCfsqp::run(patError*& err) {

  if (theProblem == NULL) {
    patString diag("No problem has been defined") ;
    return diag ;
  }

  int inform ;

  int nparam = theProblem->nVariables() ;

  int nineq = 
    theProblem->nNonLinearIneq()+
    theProblem->nLinearIneq() ;
  int neq =
    theProblem->nNonLinearEq()+
    theProblem->nLinearEq() ;


  // Lower bounds

  patReal* bl = new patReal[theProblem->nVariables()] ;
  patVariables blStl = theProblem->getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }
  copy(blStl.begin(),blStl.end(),bl) ;

  // Upper bounds

  patReal* bu = new patReal[theProblem->nVariables()] ;
  patVariables buStl = theProblem->getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }
  copy(buStl.begin(),buStl.end(),bu) ;

  // Starting point

  patReal* x = new patReal[theProblem->nVariables()] ;
  copy(startingPoint.begin(),startingPoint.end(),x) ;

  // f

  patReal* f = new patReal[1] ;

  // g
 
  int sizeg = patMax<int>(1,nineq + neq) ;
  patReal* g = new patReal[sizeg] ;

  // lamda

  int sizeLambda = theProblem->nVariables() + 1 + sizeg ;
  patReal* lambda = new patReal[sizeLambda] ;

  cfsqp(nparam                             , // nparam
	int(1)                             , // nf
	int(0)                             , // nfsr
	int(theProblem->nNonLinearIneq())  , // nineqn
	nineq                              , // nineq
	int(theProblem->nNonLinearEq())    , // neqn
	neq                                , // neq
	int(0)                             , // ncsrl
	int(0)                             , // ncsrn
	(int*)NULL                         , // mesh_pts
	mode                               , // mode
	iprint                             , // iprint
	miter                              , // miter
	&inform                            , // inform
	patReal(patMaxReal)                 , // bigbnd
	eps                                , // eps
	epseqn                             , // epseqn
	udelta                             , // udelta
	bl                                 , // bl
	bu                                 , // bu
	x                                  , // x
	f                                  , // f
	g                                  , // g
	lambda                             , // lambda
	&obj                               , // obj
	&constr                            , // constr
	&gradob                            , // gradob
	&gradcn                            , // gradcn
	(void*)theProblem                  , // cd 
	theInteraction,
	stopFile,
	&nIter) ; 
  
  stringstream str ;

  switch (inform) {
  case 0 :
    str << "Normal termination. Obj: " << eps << " Const: " << epseqn ;
    break ;
  case 1 :
  case 2 :
    str << "Unable to find a feasible strating point" ;
    break ;
  case 3 :
    str << "Maximum number of iterations " << miter << " reached" ;
    break ;
  case 4:
    str << "Line search fails. Step too small" ;
    break ;
  case 5:
  case 6:
    str << "Failure of the QP solver" ;
    break ;
  case 7:
    str << "Inconsistent input data" ;
    break ;
  case 8:
    str << "Iterations are stucked";
    break ;
  case 9:
    str << "Penalty > " << patMaxReal << ". Unable to satisfy nonlinear constraints" ;
  case 10:
    str << "Iterations interrupted by the user using file " << stopFile ;
    break ;
  default:
    str << "Unknown diagnostic" ;
  }

  // Copy solution

  for (unsigned short i = 0 ; i < nparam ; ++i) {
    solution[i] = x[i] ;
  }

  // Copy Lagrange multipliers
  
  patVariables lowB = theProblem->getLowerBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }
  patVariables upB = theProblem->getUpperBounds(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return (err->describe()) ;
  }
  unsigned int j = 0 ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    upperBoundsLambda[i] = lowerBoundsLambda[i] = 0.0 ;
    if (x[i] <= lowB[i] || patAbs(x[i]-lowB[i]) <= patEPSILON) {
      lowerBoundsLambda[i] = lambda[j] ;
    }
    else if (x[i] >= upB[i] || patAbs(x[i]-upB[i]) <= patEPSILON) {
      upperBoundsLambda[i] = lambda[j] ;
    }
    else {
      if (lambda[j] >= patEPSILON) {
	// Error: Lagrange is not zero and no bound is active for this variable
	stringstream str ;
	str << "Non zero Lagrange multiplier (" << lambda[j]
	    << ") for unactive bounds constraint:\n"
	    << lowB[i] << "<=" << x[i] << "<=" << upB[i];
	WARNING(str.str()) ;
	//	err = new patErrMiscError(str.str()) ;
	//WARNING(err->describe()) ;
	//return(err->describe()) ;
      }
    }
    ++j ;
  }
  
  theProblem->setLagrangeLowerBounds(lowerBoundsLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }
  theProblem->setLagrangeUpperBounds(upperBoundsLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }

  for (unsigned short i = 0 ; i < theProblem->nNonLinearIneq() ; ++i) {
    nonLinIneqLambda[i] = lambda[j] ;
    ++j ;
  }
  theProblem->setLagrangeNonLinIneq(nonLinIneqLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }

  for (unsigned short i = 0 ; i < theProblem->nLinearIneq() ; ++i) {
    linIneqLambda[i] = lambda[j] ;
    ++j ;
  }
  theProblem->setLagrangeLinIneq(linIneqLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }

  for (unsigned short i = 0 ; i < theProblem->nNonLinearEq() ; ++i) {
    nonLinEqLambda[i] = lambda[j] ;
    ++j ;
  }
  theProblem->setLagrangeNonLinEq(nonLinEqLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }


  for (unsigned short i = 0 ; i < theProblem->nLinearEq() ; ++i) {
    linEqLambda[i] = lambda[j] ;
    ++j ;
  }
  theProblem->setLagrangeLinEq(linEqLambda,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return(err->describe()) ;
  }

  //  DEBUG_MESSAGE("LAMBDA FOR F = " << lambda[j]) ;
  // Release allocated memory

  delete [] bl ;
  delete [] bu ;
  delete [] x ;
  delete [] f ;
  delete [] g ;
  delete [] lambda ;
  return patString(str.str()) ;
}

void obj(int nparam, int j, patReal* x, patReal* fj, void* cd) {

  patNonLinearProblem* theProblem = (patNonLinearProblem*) cd ;
  //  DEBUG_MESSAGE("Call to obj with j = " << j) ;
  trVector xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }
  patError* err = NULL ;
  patBoolean success = patTRUE ;
  trFunction* f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    *fj = patMaxReal ;
    return ;
  }
  if (f == NULL) {
    WARNING("Null pointer");
    *fj = patMaxReal ;
    return ;
  }

  *fj = f->computeFunction(&xstl,&success,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    *fj = patMaxReal ;
    return ;
  }
  if (!success) {
    *fj = patMaxReal ;
    return ;
  }
  return ;
}

void constr(int nparam,int jj,patReal* x, patReal* gj, void* cd) {  
  // cfsqp constraint number starts at 1.
  int j = jj - 1 ;
  patNonLinearProblem* theProblem = (patNonLinearProblem*) cd ;
  //  DEBUG_MESSAGE("Call to constr with j = " << j) ;

  if (gj == NULL) {
    WARNING("Null pointer.") ;
    return ;
  }
  
  patError* err = NULL;
  patBoolean success = patTRUE ;
  
  trVector xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }

  if (j < int(theProblem->nNonLinearIneq())) {
    //    DEBUG_MESSAGE("Non linear inequality") ;
    trFunction* g = theProblem->getNonLinInequality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    *gj =  g->computeFunction(&xstl,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (!success) {
      WARNING("Error in function computation") ;
      return ;
    }
    return ;
  }
  j -= theProblem->nNonLinearIneq() ;
  if (j < int(theProblem->nLinearIneq())) {
    //DEBUG_MESSAGE("Linear inequality") ;
    pair<patVariables,patReal> g = theProblem->getLinInequality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    // In the problem descrtiption, the linear equalities are Ax <= b, while
    // cfsqp requires Ax-b<=0.

    *gj = inner_product(g.first.begin(),g.first.end(),xstl.begin(),-g.second) ;
    return ;
  }
  j-= theProblem->nLinearIneq() ;

  if ( j < int(theProblem->nNonLinearEq())) {
    //    DEBUG_MESSAGE("Non linear equality" << theProblem->nNonLinearEq()) ;
    trFunction* g = theProblem->getNonLinEquality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (g == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return ;
    }
    *gj =  g->computeFunction(&xstl,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (!success) {
      WARNING("Error in function computation") ;
      return ;
    }
    return ;
  }

  j -= theProblem->nNonLinearEq() ;

  if (j < int(theProblem->nLinearEq())) {
    //    DEBUG_MESSAGE("Linear equality " << j) ;
    pair<patVariables,patReal> g = theProblem->getLinEquality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    // In the problem descrtiption, the linear equalities are Ax = b, while
    // cfsqp requires Ax-b=0.

    *gj = inner_product(g.first.begin(),g.first.end(),xstl.begin(),-g.second) ;
    return ;
  }

  WARNING("Wrong constraint number") ;
  return ;
}



void gradob(int nparam, 
	    int jj,
	    patReal* x,
	    patReal* gradfj, 
	    void (* dummy)(int, int, patReal *, patReal *, void *),
	    void* cd) {

  // cfsqp constraint number starts at 1.
  //int j = jj - 1 ;

  patNonLinearProblem* theProblem = (patNonLinearProblem*) cd ;
  //  DEBUG_MESSAGE("Call to gradob with j = " << j) ;
  trVector xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }
  patError* err = NULL ;
  patBoolean success = patTRUE ;
  trFunction* f = theProblem->getObjective(err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (f == NULL) {
    WARNING("Null pointer");
    return ;
  }

  trVector grad(xstl.size()) ;
#ifdef FINDIFF
  f->computeFinDiffGradient(&xstl,&grad,&success,err) ;
#else
  f->computeFunctionAndDerivatives(&xstl,&grad,NULL,&success,err) ;
#endif
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  if (!success) {
    WARNING("Error in function computation") ;
    return ;
  }

  copy(grad.begin(),grad.end(),gradfj) ;
  return ;
}


void gradcn(int nparam, 
	    int jj,
	    patReal* x,
	    patReal* gradgj, 
	    void (*)(int, int, patReal *, patReal *, void *),
	    void* cd) {

  int j = jj - 1 ; 
  patNonLinearProblem* theProblem = (patNonLinearProblem*) cd ;
  //  DEBUG_MESSAGE("Call to gradconstr with j = " << j) ;
  
  patError* err = NULL ;
  patBoolean success = patTRUE ;
  
  trVector xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }

  if (j < int(theProblem->nNonLinearIneq())) {
    //    DEBUG_MESSAGE("Non linear inequality") ;
    trFunction* g = theProblem->getNonLinInequality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (g == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return ;
    }
    trVector grad(xstl.size()); 
#ifdef FINDIFF
    g->computeFinDiffGradient(&xstl,&grad,&success,err) ;
#else
    g->computeFunctionAndDerivatives(&xstl,&grad,NULL,&success,err) ;
#endif
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (!success) {
      WARNING("Error in function computation") ;
      return ;
    }
    copy(grad.begin(),grad.end(),gradgj) ;
    return ;
  }
  j -= theProblem->nNonLinearIneq() ;
  if (j < int(theProblem->nLinearIneq())) {
    //   DEBUG_MESSAGE("Linear inequality") ;
    pair<patVariables,patReal> g = theProblem->getLinInequality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    copy(g.first.begin(),g.first.end(),gradgj) ;
    return ;
  }
  j-= theProblem->nLinearIneq() ;

  if ( j < int(theProblem->nNonLinearEq())) {
    //    DEBUG_MESSAGE("Non linear equality") ;
    trFunction* g = theProblem->getNonLinEquality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    trVector grad(xstl.size()) ;
    g->computeFunctionAndDerivatives(&xstl,&grad,NULL,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }
    if (!success) {
      WARNING("Error in function computation") ;
      return ;
    }
    copy(grad.begin(),grad.end(),gradgj) ;
    return ;
  }

  j -= theProblem->nNonLinearEq() ;

  if (j < int(theProblem->nLinearEq())) {
    //    DEBUG_MESSAGE("Linear equality") ;
    pair<patVariables,patReal> g = theProblem->getLinEquality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return ;
    }

    copy(g.first.begin(),g.first.end(),gradgj) ;
    return ;
  }

  WARNING("Wrong constraint number") ;
  return ;
}

patReal patCfsqp::getValueSolution(patError*& err) {
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

patString patCfsqp::getName() {
  return patString("CFSQP") ;
}
  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */

patULong patCfsqp::nbrIter() {
  return nIter ;
}

void patCfsqp::setParameters(int _mode,
			     int _iprint,
			     int _miter,
			     patReal _eps,
			     patReal _epseqn,
			     patReal _udelta,
			     patString sf) {
  
 mode = _mode ;
 iprint = _iprint ;
 miter = _miter ;
 eps = _eps ;
 epseqn = _epseqn ;
 udelta  = _udelta  ;
 stopFile = sf ;
}
