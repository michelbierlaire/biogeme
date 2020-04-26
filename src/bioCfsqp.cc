//-*-c++-*------------------------------------------------------------
//
// File name : bioCfsqp.cc
// Author :    Michel Bierlaire
// Date :      Tue Aug 13 08:57:16 2019
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <sstream>
#include <numeric>
#include "bioCfsqp.h"
#include "cfsqpusr.h"
#include "bioExceptions.h"

bioCfsqp::bioCfsqp(biogeme* bio) :
  theBiogeme(bio), 
  solution(bio->getDimension()),
  lowerBoundsLambda(bio->getDimension()),
  upperBoundsLambda(bio->getDimension()),
  lowerBounds(bio->getLowerBounds()),
  upperBounds(bio->getUpperBounds()),
  mode(100),
  iprint(2),
  miter(1000),
  eps(6.05545e-06),
  epseqn(6.05545e-06),
  udelta(0.0) ,
  nIter(static_cast<int>(0)) {

  if (lowerBounds.size() != bio->getDimension()) {
    std::stringstream str ;
    str << "Number of lower bounds ( " << lowerBounds.size() << ") inconsistent with the dimension " << bio->getDimension() ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  if (upperBounds.size() != bio->getDimension()) {
    std::stringstream str ;
    str << "Number of upper bounds ( " << upperBounds.size() << ") inconsistent with the dimension " << bio->getDimension() ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
}
 bioCfsqp::~bioCfsqp() {

}

void bioCfsqp::defineStartingPoint(const std::vector<bioReal>& x0) {
  startingPoint=x0 ;
}

std::vector<bioReal> bioCfsqp::getStartingPoint() {
  return startingPoint ;
}

std::vector<bioReal> bioCfsqp::getSolution() {
  return solution ;
}

std::vector<bioReal> bioCfsqp::getLowerBoundsLambda() {
  return lowerBoundsLambda ;
}
std::vector<bioReal> bioCfsqp::getUpperBoundsLambda() {
  return upperBoundsLambda ;
}


bioString bioCfsqp::run() {


  theBiogeme->resetFunctionEvaluations() ;
  
  int inform ;

  int nparam = theBiogeme->getDimension() ;

  int nineq = 0 ;
  int neq = 0 ; 

  // Lower bounds

  bioReal* bl = new bioReal[nparam] ;
  copy(lowerBounds.begin(),lowerBounds.end(),bl) ;

  // Upper bounds

  bioReal* bu = new bioReal[nparam] ;
  copy(upperBounds.begin(),upperBounds.end(),bu) ;

  // Starting point

  bioReal* x = new bioReal[theBiogeme->getDimension()] ;
  copy(startingPoint.begin(),startingPoint.end(),x) ;

  // f

  bioReal* f = new bioReal[1] ;

  // g
 
  //  int sizeg = patMax<int>(1,nineq + neq) ;
  int sizeg = 1 ;
  bioReal* g = new bioReal[sizeg] ;

  // lamda

  int sizeLambda = theBiogeme->getDimension() + 1 + sizeg ;
  bioReal* lambda = new bioReal[sizeLambda] ;

  
  cfsqp(nparam                             , // nparam
	int(1)                             , // nf
	int(0)                             , // nfsr
	int(0)                             , // nineqn
	0                                  , // nineq
	int(0)                             , // neqn
	0                                  , // neq
	int(0)                             , // ncsrl
	int(0)                             , // ncsrn
	(int*)NULL                         , // mesh_pts
	mode                               , // mode
	iprint                             , // iprint
	miter                              , // miter
	&inform                            , // inform
	bioReal(bioMaxReal)                , // bigbnd
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
	(void*)theBiogeme                  , // cd
	&nIter) ; 
  
  std::stringstream str ;

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
    str << "Penalty > " << bioMaxReal << ". Unable to satisfy nonlinear constraints" ;
  case 10:
    str << "Iterations interrupted by the user";
    break ;
  default:
    str << "Unknown diagnostic" ;
  }

  // Copy solution
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    solution[i] = x[i] ;
  }

  // Copy Lagrange multipliers
  
  unsigned int j = 0 ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    upperBoundsLambda[i] = lowerBoundsLambda[i] = 0.0 ;
    if (x[i] <= lowerBounds[i] || std::abs(x[i]-lowerBounds[i]) <= bioEpsilon) {
      lowerBoundsLambda[i] = lambda[j] ;
    }
    else if (x[i] >= upperBounds[i] || std::abs(x[i]-upperBounds[i]) <= bioEpsilon) {
      upperBoundsLambda[i] = lambda[j] ;
    }
    else {
      if (lambda[j] >= bioEpsilon) {
	// Error: Lagrange is not zero and no bound is active for this variable
	std::stringstream str ;
	str << "Non zero Lagrange multiplier (" << lambda[j]
	    << ") for unactive bounds constraint:\n"
	    << lowerBounds[i] << "<=" << x[i] << "<=" << upperBounds[i];
	WARNING(str.str()) ;
	//	err = new patErrMiscError(str.str()) ;
	//WARNING(err->describe()) ;
	//return(err->describe()) ;
      }
    }
    ++j ;
  }
  

  //  DEBUG_MESSAGE("LAMBDA FOR F = " << lambda[j]) ;
  // Release allocated memory

  delete [] bl ;
  delete [] bu ;
  delete [] x ;
  delete [] f ;
  delete [] g ;
  delete [] lambda ;
  return bioString(str.str()) ;
}

void obj(int nparam, int j, bioReal* x, bioReal* fj, void* cd) {

  biogeme* theBiogeme = (biogeme*) cd ;
  //  DEBUG_MESSAGE("Call to obj with j = " << j) ;
  std::vector<bioReal> xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }
  //  try {
  *fj = -theBiogeme->repeatedCalculateLikelihood(xstl) ;
  //  }
  //  catch(...) {
  //    *fj = patMaxReal ;
  //  }
  return ;
}

void constr(int nparam,int jj,bioReal* x, bioReal* gj, void* cd) {
  std::stringstream str ;
  str << "This function should not be called. The problem has no constraint" ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  return ;
}

void gradob(int nparam, 
	    int jj,
	    bioReal* x,
	    bioReal* gradfj, 
	    void (* dummy)(int, int, bioReal *, bioReal *, void *),
	    void* cd) {

  // cfsqp constraint number starts at 1.
  //int j = jj - 1 ;

  biogeme* theBiogeme = (biogeme*) cd ;
  std::vector<bioReal> xstl(nparam) ;
  for (unsigned short i = 0 ; i < nparam ; ++i) {
    xstl[i] = x[i] ;
  }

  std::vector<bioReal> grad(xstl.size()) ;
  std::vector< std::vector<bioReal> > hh ;
  theBiogeme->repeatedCalcLikeAndDerivatives(xstl,grad,hh,hh,false,false) ;
  for (bioUInt i = 0 ; i < xstl.size() ; ++i) {
    gradfj[i] = -grad[i] ;
  }
  return ;
}


void gradcn(int nparam, 
	    int jj,
	    bioReal* x,
	    bioReal* gradgj, 
	    void (*)(int, int, bioReal *, bioReal *, void *),
	    void* cd) {

  std::stringstream str ;
  str << "This function should not be called. The problem has no constraint" ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  return ;
}

bioReal bioCfsqp::getValueSolution() {
  bioReal result = -theBiogeme->repeatedCalculateLikelihood(solution) ;
  return result ;
}

  /**
     @return number of iterations. If there is any error, 0 is returned. 
   */

bioUInt bioCfsqp::nbrIter() {
  return nIter ;
}

void bioCfsqp::setParameters(int _mode,
			     int _iprint,
			     int _miter,
			     bioReal _eps,
			     bioReal _epseqn,
			     bioReal _udelta) {
  
 mode = _mode ;
 iprint = _iprint ;
 miter = _miter ;
 eps = _eps ;
 epseqn = _epseqn ;
 udelta  = _udelta  ;
}
