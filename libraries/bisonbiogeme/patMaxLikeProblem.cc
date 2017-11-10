//-*-c++-*------------------------------------------------------------
//
// File name : patMaxLikeProblem.cc
// Author :    Michel Bierlaire
// Date :      Fri Apr 27 11:39:51 2001
//
//--------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <fstream>
#include <numeric>
#include <assert.h>
#include <sstream>


#include "patMath.h"
#include "patParameters.h"
#include "patArithNode.h"
#include "patLikelihood.h"
#include "patModelSpec.h"
#include "patErrNullPointer.h"
#include "patErrMiscError.h"
#include "patMaxLikeProblem.h"
#include "patErrOutOfRange.h"
#include "trFunction.h"
#include "trBounds.h" 
#include "patTimeInterval.h"
#include "patInverseIteration.h"
#include "patSvd.h"

patMaxLikeProblem::patMaxLikeProblem(trFunction* _f, trBounds* _b, trParameters p) :
  f(_f) ,
  bounds(_b),
  svd(NULL),
  bigMatrix(NULL) ,
  matHess(NULL),
  theTrParameters(p)
 {
}

patMaxLikeProblem::~patMaxLikeProblem() {
  DELETE_PTR(svd) ;
  DELETE_PTR(bigMatrix) ;
  DELETE_PTR(matHess) ;
}
  
unsigned long patMaxLikeProblem::nVariables() {
  return ((f == NULL)
	  ?0
	  :f->getDimension()) ;
}

unsigned long patMaxLikeProblem::nNonLinearIneq() {
  return 0 ;
}
unsigned long patMaxLikeProblem::nNonLinearEq() {
  return nonLinEqConstraints.size() ;
}

unsigned long patMaxLikeProblem::nLinearIneq() {
  return linIneqConstraints.size() ;
}

unsigned long patMaxLikeProblem::nLinearEq() {
  return linEqConstraints.size() ;
}

patVariables patMaxLikeProblem::getLowerBounds(patError*& err) {
  if (bounds == NULL) {
    patVariables lb(nVariables(),-patMaxReal) ;
    return lb ;
  }
  return bounds->getLowerVector() ;
}

patVariables patMaxLikeProblem::getUpperBounds(patError*& err) {
  if (bounds == NULL) {
    patVariables ub(nVariables(),patMaxReal) ;
    return ub ;
  }
  return bounds->getUpperVector() ;
}

trFunction* patMaxLikeProblem::getObjective(patError*& err) {
  return f ;
}

trFunction* patMaxLikeProblem::getNonLinInequality(unsigned long i,
						 patError*& err) {
  return NULL ;
}

trFunction* patMaxLikeProblem::getNonLinEquality(unsigned long i,
					       patError*& err) {
  if (i >= nNonLinearEq()) {
    err = new patErrOutOfRange<unsigned long>(i,0,nNonLinearEq()-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  } 
  return nonLinEqConstraints[i] ;
}

pair<patVariables,patReal> patMaxLikeProblem::getLinInequality(unsigned long i,
							     patError*& err) {
  if (i >= nLinearIneq()){
    err = new patErrOutOfRange<unsigned long>(i,0,nLinearIneq()-1) ;
    WARNING(err->describe()) ;
    return pair<patVariables,patReal>() ;
  } 
  return linIneqConstraints[i] ;
}

pair<patVariables,patReal> patMaxLikeProblem::getLinEquality(unsigned long i,
							   patError*& err) {
  if (i >= nLinearEq()){
    err = new patErrOutOfRange<unsigned long>(i,0,nLinearEq()-1) ;
    WARNING(err->describe()) ;
    return pair<patVariables,patReal>() ;
  } 
  return linEqConstraints[i] ;

}
  
int patMaxLikeProblem::addNonLinEq(trFunction* f) {
  int rank = nonLinEqConstraints.size() ;
  nonLinEqConstraints.push_back(f) ;
  return rank ;
}

int patMaxLikeProblem::addNonLinIneq(trFunction* f) {
  WARNING("Sorry. No nonlinear inequality is allowed at this time") ;
  exit(-1) ;
}

int patMaxLikeProblem::addLinEq(const patVariables& a, patReal b) {
  //  DEBUG_MESSAGE("Add const" << a) ;
  //DEBUG_MESSAGE("b = " << b) ;
  int rank = linEqConstraints.size() ;
  linEqConstraints.push_back(pair<patVariables,patReal>(a,b)) ;
  return rank ;
}

int patMaxLikeProblem::addLinIneq(const patVariables& a, patReal b) {
  //DEBUG_MESSAGE("Add inequality const" << a) ;
  //DEBUG_MESSAGE("b = " << b) ;
  int rank = linIneqConstraints.size() ;
  linIneqConstraints.push_back(pair<patVariables,patReal>(a,b)) ;
  return rank ;
}




patBoolean patMaxLikeProblem::computeVarCovar(trVector* x,
					       patMyMatrix* varCovar,
					       patMyMatrix* robustVarCovar,
					      map<patReal,patVariables>* eigVec,
					      patReal* smallSV,
					       patError*& err) {

  //Reference: Schoenberg, R. (1997) Constrained Maximum Likelihood,
  //Computational Economics, 10: 251-266.

  // if (!f->isHessianAvailable()) {
  //   WARNING("Hessian is not available. Unable to compute var-covar matrix") ;
  //   return patFALSE ;
  // }

  patBoolean computeRobust = (robustVarCovar != NULL) ;

  if (varCovar == NULL) {
    err = new patErrNullPointer("patMyMatrix") ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  trHessian hess(theTrParameters,nVariables()) ;
  trHessian bhhh(theTrParameters,nVariables()) ;
  patAbsTime b ;
  patAbsTime e ;
  //  DEBUG_MESSAGE("Compute hessian") ;
  b.setTimeOfDay() ;
  patBoolean success(patTRUE) ;

  ofstream yyyy("hess.lis") ;

  if (computeRobust) {
    if (robustVarCovar->nRows() != nVariables() || 
	robustVarCovar->nCols() != nVariables()) {
      stringstream str ;
      str << "robustVarCovar should be " << nVariables() << "x" << nVariables() << " and not " << robustVarCovar->nRows() << " * " << robustVarCovar->nCols() ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }

    patVariables grad(nVariables()) ;
    // Compute the gradient, and build BHHH 
    f->computeFunctionAndDerivatives(x,&grad,NULL,&success,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }  
    if (!success) {
      WARNING("Unsuccessful computation of BHHH.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
    
    f->computeCheapHessian(&bhhh,err) ;
    //    DEBUG_MESSAGE("BHHH = " << bhhh) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }

    yyyy << "============== BHHH ===============" << endl ;
    yyyy << bhhh << endl ;
    
    if (!success) {
      WARNING("Unsuccessful computation of BHHH.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
  }
  
  if (patParameters::the()->getgevVarCovarFromBHHH() == 0) { 

    //    DEBUG_MESSAGE("Compute hessian") ;
    if (f->isHessianAvailable()) {
      patVariables grad(nVariables()) ;
      f->computeFunctionAndDerivatives(x,&grad,&hess,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return patFALSE;
      }  
    }
    else {
      f->computeFinDiffHessian(x,&hess,&success,err) ;
    }
    //    DEBUG_MESSAGE("Hessian = " << hess) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }
    //    DEBUG_MESSAGE("Done.") ;
    e.setTimeOfDay() ;
    patTimeInterval ti0(b,e) ;
    //    DEBUG_MESSAGE("Compute hessian in " << ti0.getLength()) ;

    yyyy << "============== Hessian ===============" << endl ;
    yyyy << hess << endl ;

    if (!success) {
      WARNING("Unsuccessful Hessian computation.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
  
  }
  else {
    hess = bhhh ;
  }

  if (lagrangeNonLinEqConstraints.size() != nNonLinearEq()) {
    stringstream str ;
    str << lagrangeNonLinEqConstraints.size() << " Lagrange multipliers for " << nNonLinearEq() << " non linear equality constraints" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  for (unsigned long j = 0 ; j < nNonLinearEq() ; ++j) {
    trFunction* g = getNonLinEquality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    if (g == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    
    if (!g->isHessianAvailable()) {
      WARNING("Constraint Hessian is not available. Unable to compute var-covar matrix") ;
      return patFALSE ;
    }

    trHessian constrHess(theTrParameters,nVariables()) ;    
    g->computeFinDiffHessian(x,&constrHess,&success,err) ;
    if (!success) {
      WARNING("Unsuccessful constraint's Hessian computation.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }
    hess.add(lagrangeNonLinEqConstraints[j],constrHess,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }
  }
  
  if (lagrangeNonLinIneqConstraints.size() != nNonLinearIneq()) {
    stringstream str ;
    str << lagrangeNonLinIneqConstraints.size() << " Lagrange multipliers for " << nNonLinearIneq() << " non linear inequality constraints" ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  for (unsigned long j = 0 ; j < nNonLinearIneq() ; ++j) {
    trFunction* h = getNonLinInequality(j,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    if (h == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    
    if (!h->isHessianAvailable()) {
      WARNING("Hessian is not available. Unable to compute var-covar matrix") ;
      return patFALSE ;
    }

    trHessian constrHess(theTrParameters,nVariables()) ;
    h->computeFinDiffHessian(x,&constrHess,&success,err) ;
    if (!success) {
      WARNING("Unsuccessful constraint's Hessian computation.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }
    hess.add(lagrangeNonLinIneqConstraints[j],constrHess,err) ;
    if (err != NULL) {
      WARNING(err->describe());
      return patFALSE;
    }
  }
  matHess = new patMyMatrix (hess.getMatrixForLinAlgPackage(err)) ;
  if (err != NULL) {
    WARNING(err->describe());
    return patFALSE;
  }

//    DEBUG_MESSAGE("Hessian=") ;
//    cout << matHess << endl ;
  // Build the augmented matrix
  
//    DEBUG_MESSAGE("nVariables = " << nVariables()) ;
//    DEBUG_MESSAGE("nLinearEq = " << nLinearEq()) ;
//    DEBUG_MESSAGE("nLinearIneq = " << nLinearIneq()) ;
//    DEBUG_MESSAGE("nNonLinearEq = " << nNonLinearEq()) ;
//    DEBUG_MESSAGE("nNonLinearIneq = " << nNonLinearIneq()) ;
  
//    DEBUG_MESSAGE("nConstraints = " << nConstraints()) ;
  unsigned long nbrOfSlacks = 
    2*nVariables() + nLinearIneq() + nNonLinearIneq() ;
  
  unsigned long size = 
    5*nVariables() + 2 * nLinearIneq() + 2 * nNonLinearIneq() 
    + nLinearEq() + nNonLinearEq() ;
  
  if (varCovar->nRows() != size || varCovar->nCols() != size) {
    stringstream str ;
    str << "varCovar should be " << size << "x" << size << " and not " << varCovar->nRows() << " * " << varCovar->nCols() ;
    err = new patErrMiscError(str.str()) ;
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  bigMatrix = new patMyMatrix(size,size) ;
  for (unsigned long kk = 0 ; kk < size ; ++kk) {
    for (unsigned long ll = 0 ; ll < size ; ++ll) {
      (*bigMatrix)[kk][ll] = 0.0 ;
    }
  }

  //  DEBUG_MESSAGE("bigmatrix size = " << size) ;
  
  for (unsigned long kk = 0 ; kk < nVariables() ; ++kk) {
    for (unsigned long ll = 0 ; ll < nVariables() ; ++ll) {
      (*bigMatrix)[kk][ll] = (*matHess)[kk][ll] ;
    }
  }
  
  //Bounds
  for (unsigned long i = 0 ; i < nVariables() ; ++i) {
    //Lower
    
    (*bigMatrix)[i][nVariables()+nbrOfSlacks+i] = -1.0 ;
    (*bigMatrix)[nVariables()+nbrOfSlacks+i][i] = -1.0 ;
    // Upper
    (*bigMatrix)[i][2*nVariables()+nbrOfSlacks+i] = 1.0 ;
    (*bigMatrix)[2*nVariables()+nbrOfSlacks+i][i] = 1.0 ;
    // Slacks lower
    patReal lowerB = bounds->getLower(i,err) ;
    if (err != NULL) {
      err->describe() ;
      return patFALSE ;
    }
    if ((*x)[i] - lowerB <= -sqrt(patEPSILON)) {
      stringstream str ;
      str << "x(" << i << ")=" << (*x)[i] << " less then lower bound " << lowerB ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patFALSE;
    }
    
    (*bigMatrix)[nVariables()+i][nVariables()+nbrOfSlacks+i] =
      (*bigMatrix)[nVariables()+nbrOfSlacks+i][nVariables()+i] =
      2.0 * sqrt(patMax(patZero,(*x)[i]-lowerB)) ;

    
    // Slacks upper
    patReal upperB = bounds->getUpper(i,err) ;
    if (err != NULL) {
      err->describe() ;
      return patFALSE ;
    }
    if (upperB - (*x)[i] <= -sqrt(patEPSILON)) {
      stringstream str ;
      str << "x(" << i << ")=" << (*x)[i] << " greater then upper bound " << upperB ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patFALSE;
    }
    (*bigMatrix)[2*nVariables()+i][2*nVariables()+nbrOfSlacks+i] =
      (*bigMatrix)[2*nVariables()+nbrOfSlacks+i][2*nVariables()+i] =
      2.0 * sqrt(patMax(patZero,upperB-(*x)[i])) ;
  }
  
  //  (*bigMatrix).print("Bounds") ;
  //Lagrange

  unsigned long start = nVariables() ;
  for (unsigned long j = 0 ; j < lagrangeLowerBounds.size() ; ++j) {
    (*bigMatrix)[start+j][start+j] = 2.0 * lagrangeLowerBounds[j] ;
  }
  start += nVariables() ;
  for (unsigned long j = 0 ; j < lagrangeUpperBounds.size() ; ++j) {
    (*bigMatrix)[start+j][start+j] = 2.0 * lagrangeUpperBounds[j] ;
  }
  start += nVariables() ;
  for (unsigned long j = 0 ; j < lagrangeLinIneqConstraints.size() ; ++j) {
    (*bigMatrix)[start+j][start+j] = 2.0 * lagrangeLinIneqConstraints[j] ;
  }
  start += nLinearIneq() ;
  for (unsigned long j = 0 ; j < lagrangeNonLinIneqConstraints.size() ; ++j) {
    (*bigMatrix)[start+j][start+j] = 2.0 * lagrangeNonLinIneqConstraints[j] ;
  }

  //  (*bigMatrix).print("Lagrange") ;
  //Linear equality
  start = 5*nVariables() + nLinearIneq() + nNonLinearIneq() ;
  for (unsigned long i = 0 ; i < nLinearEq() ; ++i) {
    pair<patVariables,patReal> c = getLinEquality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    for (unsigned long j = 0 ; j < nVariables() ; ++j) {
      (*bigMatrix)[j][i+start] = 
	(*bigMatrix)[i+start][j] = c.first[j] ;
    }
  } 
  
  //  bigMatrix.print("Linear equality") ;
  //Linear inequality
  start += nLinearEq() ;
  for (unsigned long i = 0 ; i < nLinearIneq() ; ++i) {
    pair<patVariables,patReal> c = getLinInequality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    for (unsigned long j = 0 ; j < nVariables() ; ++j) {
      (*bigMatrix)[j][i+start] = 
	(*bigMatrix)[i+start][j] = c.first[j] ;
    }

    // Slack variable

    patReal cx = inner_product(c.first.begin(),c.first.end(),x->begin(),0.0) ;

    if (c.second - cx <= -patEPSILON) {
      stringstream str ;
      str << "c'x = " << cx << " should be less than d = " 
	  << c.second ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patFALSE;
    }

    (*bigMatrix)[3*nVariables()+i][i+start] =
      (*bigMatrix)[i+start][3*nVariables()+i] =
      2.0 * sqrt(patMax(patZero,c.second - cx)) ;
  } 
      
  //  bigMatrix.print("Linear inequality") ;
  // Non linear equality
  start += nLinearIneq() ;
  for (unsigned long i = 0 ; i < nNonLinearEq() ; ++i) {
    trFunction* c = getNonLinEquality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (c == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return patFALSE;
    }
    patVariables deriv(c->getDimension()) ;
    c->computeFunctionAndDerivatives(x,
				     &deriv,
				     NULL,
				     &success,
				     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (!success) {
      err = new patErrMiscError("Error in constraint gradient computation") ;
      WARNING(err->describe()) ;
      return patFALSE;
    }
    for (unsigned long j = 0 ; j < nVariables() ; ++j) {
      (*bigMatrix)[j][i+start] = 
	(*bigMatrix)[i+start][j] = deriv[j] ;
    }
  }
      
  //  bigMatrix.print("Non linear equality") ;
  // Non linear inequality
  start += nNonLinearEq() ;
  for (unsigned long i = 0 ; i < nNonLinearIneq() ; ++i) {
    trFunction* c = getNonLinInequality(i,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (c == NULL) {
      err = new patErrNullPointer("trFunction") ;
      WARNING(err->describe()) ;
      return patFALSE;
    }

    patVariables deriv(x->size()) ;
    c->computeFunctionAndDerivatives(x,
				     &deriv,
				     NULL,
				     &success,
				     err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (!success) {
      err = new patErrMiscError("Error in constraint gradient computation") ;
      WARNING(err->describe()) ;
      return patFALSE;
    }
    for (unsigned long j = 0 ; j < nVariables() ; ++j) {
      (*bigMatrix)[j][i+start] = 
	(*bigMatrix)[i+start][j] = deriv[j] ;
    }

    // Slack variable

    patReal hx = c->computeFunction(x,
				&success,
				err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (!success) {
      err = new patErrMiscError("Error in constraint gradient computation") ;
      WARNING(err->describe()) ;
      return patFALSE;
    }

    if (hx > patEPSILON) {
      stringstream str ;
      str << "h(x) = " << hx << " and should be negative" ;
      err = new patErrMiscError(str.str()) ;
      WARNING(err->describe()) ;
      return patFALSE ;
    }
    (*bigMatrix)[3*nVariables()+nLinearIneq()+i][i+start] =
      (*bigMatrix)[i+start][3*nVariables()+i] =
      2.0 * sqrt(patMax(patZero,-hx)) ;

      
  }

  //  vector<int> pvector(bigMatrix.nRows());

  DEBUG_MESSAGE("... Inverse the projected hessian [" << size << "x" << size <<"]") ;

  svd = new patSvd(bigMatrix) ;
  DEBUG_MESSAGE("... Compute the singular value decomposition") ;
  svd->computeSvd(patParameters::the()->getsvdMaxIter(),
		  err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  const patMyMatrix* sol = svd->computeInverse() ;

  if (sol == NULL) {
    DEBUG_MESSAGE("Unable to compute the SVD") ;
    return patFALSE ;
  }
  *varCovar = *sol ;
  *eigVec = svd->getEigenVectorsOfZeroEigenValues(patParameters::the()->getgevSingularValueThreshold()) ;
  *smallSV = svd->getSmallestSingularValue() ;
  //  DEBUG_MESSAGE(*svd->getSingularValues()) ;
  DEBUG_MESSAGE("*******************************") ;
  DEBUG_MESSAGE("There are " << eigVec->size() << " zeros singular values") ;
  DEBUG_MESSAGE("*******************************") ;
  
  //   if (!inverse.isInvertible()) {
  //      WARNING("Matrix not invertible") ;
  //      return patFALSE ;
  //   }
  
  
  // Compute the robust var covar using the "sandwich" estimator
  DEBUG_MESSAGE("... Compute the robust var covar using the \"sandwich\" estimator") ;
  
  if (computeRobust) {
    
    patMyMatrix matBhhh(bhhh.getMatrixForLinAlgPackage(err)) ;
//     DEBUG_MESSAGE("+++++++++++++++++") ;
//     DEBUG_MESSAGE("bhhh=" << matBhhh) ;
//     DEBUG_MESSAGE("varcovar=" << (*varCovar)) ;
    patMyMatrix tmp(nVariables(),nVariables()) ;
    patMyMatrix restrVarCovar(nVariables(),nVariables()) ;
    for (unsigned long i = 0 ; i < nVariables() ; ++i) {
      for (unsigned long j = 0 ; j < nVariables() ; ++j) {
	restrVarCovar[i][j] = (*varCovar)[i][j] ;
      }
    }
    mult(matBhhh,restrVarCovar,tmp,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    mult(restrVarCovar,tmp,*robustVarCovar,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    //    DEBUG_MESSAGE("robustvarcovar=" << (*robustVarCovar)) ;
  }
  
  DEBUG_MESSAGE("... Done.") ;
  return patTRUE ;

}



patString patMaxLikeProblem::getProblemName() {
  patString name = "Max. likelihood for " ;
  if (patModelSpec::the()->isMNL()) {
    name += "logit" ;
  }
  if (patModelSpec::the()->isNL()) {
    name += "nested logit" ;
  }
  if (patModelSpec::the()->isCNL()) {
    name += "cross-nested logit" ;
  }
  return name ;
}







patBoolean patMaxLikeProblem::isFeasible(trVector& x, patError*& err) const {
  
  patBoolean result = patTRUE ;

  if (bounds == NULL) {
    err = new patErrNullPointer("trBounds") ;
  }
  patBoolean verifyBounds = bounds->isFeasible(x,err) ;
  if (!verifyBounds) {
    result = patFALSE ;
    WARNING("Bounds constraints are not verified") ;
  }

  for (unsigned long i = 0 ; i < linEqConstraints.size() ; ++i) {
    patReal lhs = inner_product(x.begin(),
				x.end(),
				linEqConstraints[i].first.begin(),0.0) ;
    patReal rhs = linEqConstraints[i].second ;

    if (patAbs(lhs-rhs) >= patEPSILON) {
      WARNING("Lin. equality " << i << " not verified. Lhs = " << lhs
	      << ", rhs = " << rhs) ;
      DEBUG_MESSAGE("c = " << patVariables(linEqConstraints[i].first)) ;
      DEBUG_MESSAGE("x = " << x) ;
      result = patFALSE ;
    }
  }
  
  for (unsigned long i = 0 ; i < nonLinEqConstraints.size() ; ++i) {
    trFunction* h = nonLinEqConstraints[i] ;
    if (h == NULL) {
      err = new patErrNullPointer("trFunction") ;
      return patFALSE ;
    }
    patBoolean success ;
    patReal hx = h->computeFunction(&x,&success, err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return patFALSE;
    }
    if (!success) {
      WARNING("Unable to evaluate non linear eq. constraint " << i) ;
      result = patFALSE ;
    }
    if (patAbs(hx) > patEPSILON) {
      WARNING("Non linear eq. " << i << " not verifed [" << hx << "]") ;
      result = patFALSE ;
    }

  }
  return result ;
}

unsigned long patMaxLikeProblem::getSizeOfVarCovar() {
  return(5*nVariables() + 2 * nLinearIneq() + 2 * nNonLinearIneq() 
	 + nLinearEq() + nNonLinearEq()) ;
  
}
