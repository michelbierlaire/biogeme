//-*-c++-*------------------------------------------------------------
//
// File name : bioMinimizationProblem.cc
// Author :    Michel Bierlaire and Mamy Fetiarison
// Date :      Fri Aug  7 10:23:45 2009
//
//--------------------------------------------------------------------

#include <numeric>

#include "bioMinimizationProblem.h"
#include "bioParameters.h"
#include "patTimeInterval.h"
#include "trFunction.h"
#include "patErrOutOfRange.h"
#include "patErrNullPointer.h"
#include "patMath.h"
#include "patSvd.h"

bioMinimizationProblem::bioMinimizationProblem(trFunction* f,trBounds b, trParameters p):
  theFunction(f), theBounds(b) , theTrParameters(p) {
  
  f->setStopFileName(bioParameters::the()->getValueString("stopFileName")) ;
}
unsigned long bioMinimizationProblem::nVariables() {
  return ((theFunction == NULL)
	  ?0
	  :theFunction->getDimension()) ;
}

unsigned long bioMinimizationProblem::nNonLinearIneq() {
  return 0 ;
}

unsigned long bioMinimizationProblem::nNonLinearEq() {
  return equalityConstraints.size() ;
}

unsigned long bioMinimizationProblem::nLinearIneq() {
  return 0 ;
}

unsigned long bioMinimizationProblem::nLinearEq() {
  return 0 ;
}

patVariables bioMinimizationProblem::getLowerBounds(patError*& err) {
  return theBounds.getLowerVector() ;

}
patVariables bioMinimizationProblem::getUpperBounds(patError*& err) {
  return theBounds.getUpperVector() ;
}

trFunction* bioMinimizationProblem::getObjective(patError*& err) {
  return theFunction ;
}

trFunction* bioMinimizationProblem::getNonLinInequality(unsigned long i,
							patError*& err) {
  return NULL ;
 
}

trFunction* bioMinimizationProblem::getNonLinEquality(unsigned long i,
						      patError*& err) {
  if (i >= equalityConstraints.size()) {
    err = new patErrOutOfRange<patULong>(i,0,equalityConstraints.size()-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return equalityConstraints[i] ;
}

pair<patVariables,patReal> 
bioMinimizationProblem::getLinInequality(unsigned long i,
		 patError*& err) {
  return pair<patVariables,patReal>() ;
}

pair<patVariables,patReal> 
bioMinimizationProblem::getLinEquality(unsigned long i,
				       patError*& err) {
  return pair<patVariables,patReal>() ;
}

patString bioMinimizationProblem::getProblemName() {
  patString name("Maximum likelihood problem for biogeme") ;
  return name ;
}

void bioMinimizationProblem::addEqualityConstraint(trFunction* c) {
  if (c == NULL) {
    WARNING("NUll pointer") ;
    return ;
  }
  equalityConstraints.push_back(c) ;
}





patBoolean bioMinimizationProblem::isFeasible(trVector& x, patError*& err) const {
  
  patBoolean result = patTRUE ;
  
  patBoolean verifyBounds = theBounds.isFeasible(x,err) ;
  if (!verifyBounds) {
    result = patFALSE ;
    WARNING("Bounds constraints are not verified") ;
  }
  
  for (unsigned long i = 0 ; i < equalityConstraints.size() ; ++i) {
    trFunction* h = equalityConstraints[i] ;
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

patBoolean bioMinimizationProblem::computeVarCovar(trVector* x,
						   patMyMatrix* varCovar,
						   patMyMatrix* robustVarCovar,
						   map<patReal,patVariables>* eigVec,
						   patReal* smallSV,
						   patError*& err) {
  
  //Reference: Schoenberg, R. (1997) Constrained Maximum Likelihood,
  //Computational Economics, 10: 251-266.

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
    if (theFunction->isHessianAvailable()) {
      
      theFunction->computeFunctionAndDerivatives(x,&grad,&hess,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return patFALSE;
      }  
    }
    else {
      theFunction->computeFunctionAndDerivatives(x,&grad,NULL,&success,err) ;
      if (err != NULL) {
	WARNING(err->describe());
	return patFALSE;
      }  
      // theFunction->computeFinDiffHessian(x,&hess,&success,err) ;
      // if (err != NULL) {
      // 	WARNING(err->describe());
      // 	return patFALSE;
      // }  

    }
    if (!success) {
      WARNING("Unsuccessful computation of BHHH.  Unable to compute var-covar matrix") ;
      return patFALSE ;
    }
    
    theFunction->computeCheapHessian(&bhhh,err) ;
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
  
  patBoolean varCovarFromBHHH = bioParameters::the()->getValueInt("varCovarFromBHHH",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  patBoolean analyticalHessian = bioParameters::the()->getValueInt("deriveAnalyticalHessian",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  if (varCovarFromBHHH == 0) {

    //    DEBUG_MESSAGE("Compute hessian") ;
    if (!analyticalHessian) {
      theFunction->computeFinDiffHessian(x,&hess,&success,err) ;
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

    vector<patReal> constrGrad(nVariables()) ;
    trHessian constrHess(theTrParameters,nVariables()) ;    
    g->computeFunctionAndDerivatives(x,&constrGrad,&constrHess,&success,err) ;
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

    vector<patReal> constrGrad(nVariables()) ;
    trHessian constrHess(theTrParameters,nVariables()) ;
    h->computeFunctionAndDerivatives(x,&constrGrad,&constrHess,&success,err) ;
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
  matHess = new patMyMatrix(hess.getMatrixForLinAlgPackage(err)) ;
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
    patReal lowerB = theBounds.getLower(i,err) ;
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
      2.0 * sqrt(patMax(0.0,(*x)[i]-lowerB)) ;

    
    // Slacks upper
    patReal upperB = theBounds.getUpper(i,err) ;
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
      2.0 * sqrt(patMax(0.0,upperB-(*x)[i])) ;
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
      2.0 * sqrt(patMax(0.0,c.second - cx)) ;
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
      2.0 * sqrt(patMax(0.0,-hx)) ;

      
  }

  //  vector<int> pvector(bigMatrix.nRows());

  DEBUG_MESSAGE("... Inverse the projected hessian [" << size << "x" << size <<"]") ;

  patSvd svd(bigMatrix) ;
  DEBUG_MESSAGE("... Compute the singular value decomposition") ;
  patULong svdMaxIter = bioParameters::the()->getValueInt("svdMaxIter",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  svd.computeSvd(svdMaxIter,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }
  const patMyMatrix* sol = svd.computeInverse() ;

  if (sol == NULL) {
    DEBUG_MESSAGE("Unable to compute the SVD") ;
    return patFALSE ;
  }
  else {
    *varCovar = *sol ;
  }

  patReal svdThreshold = bioParameters::the()->getValueReal("singularValueThreshold",err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patFALSE ;
  }

  *eigVec = svd.getEigenVectorsOfZeroEigenValues(svdThreshold) ;
  *smallSV = svd.getSmallestSingularValue() ;
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
    //    DEBUG_MESSAGE("+++++++++++++++++") ;
    // DEBUG_MESSAGE("bhhh=" << matBhhh) ;
    // DEBUG_MESSAGE("n = " << nVariables()) ;
    //DEBUG_MESSAGE("varcovar=" << (*varCovar)) ;
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
    // DEBUG_MESSAGE("robustvarcovar=" << (*robustVarCovar)) ;
  }
  
  DEBUG_MESSAGE("... Done.") ;
  return patTRUE ;

}

patULong bioMinimizationProblem::getSizeOfVarCovar() {
  return(5*nVariables() + 2 * nLinearIneq() + 2 * nNonLinearIneq() 
	 + nLinearEq() + nNonLinearEq()) ;
  
}
