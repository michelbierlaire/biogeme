//-*-c++-*------------------------------------------------------------
//
// File name : trTointSteihaug.cc
// Author :    Michel Bierlaire
// Date :      Wed Jan 19 15:04:32 2000
//
// Implementation of algorithme 7.5.1: the Steihaug-Toint truncated CG 
// method
//
// Source: Conn, Gould Toint (2000) Trust Region Methods
//--------------------------------------------------------------------

#include <numeric>
#include <cmath>
#include "patDisplay.h"
#include "patMath.h"
#include "patErrNullPointer.h"
#include "trTointSteihaug.h"
#include "trVector.h"
#include "trMatrixVector.h"
#include "trPrecond.h"

trTointSteihaug::trTointSteihaug(const  trVector* _g,
				 trMatrixVector* _H,
				 patReal _radius,
				 const trPrecond* _m,
				 trParameters p,
				 patError*& err) :
  g(_g),
  normMsk2(0.0),
  H(_H),
  M(_m),
  radius(_radius),
  status(trUNKNOWN),
  solution(g->size()),
  theParameters(p),
  statusName(5){


  normG = sqrt(inner_product(g->begin(),g->end(),g->begin(),0.0)) ;
  kfgr = theParameters.fractionGradientRequired ;
  theta = theParameters.expTheta ;
  maxIter = 5*g->size() ;
  
  statusName[trUNKNOWN] =   "Unknown " ;
  statusName[trNEG_CURV] =  "NegCurv " ;
  statusName[trOUT_OF_TR] = "OutTrReg" ;
  statusName[trMAXITER] =   "MaxIter " ;
  statusName[trCONV] =      "Converg " ;
}
				   
trTointSteihaug::trTermStatus trTointSteihaug::getTermStatus(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }
  return status ;
}

trTointSteihaug::trTermStatus trTointSteihaug::run(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }
  
  fill(solution.begin(),solution.end(),0.0) ;

  if (g == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }

  trVector gk(*g) ;

  if (M == NULL) {
    err = new patErrNullPointer("trVector") ;
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }

  patReal gkNorm = normG ;

  trVector vk = M->solve(&gk,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }
  trVector pk = -vk ;

  int iter = 0 ;

  if (H == NULL) {
    err = new patErrNullPointer("trMatrixVector") ;
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }

  patReal gkvk = inner_product(gk.begin(),gk.end(),vk.begin(),0.0) ;

  patReal skMpk = 0.0 ;
  patReal normMpk2 = gkvk ;
  normMsk2 = 0.0 ;

  while (gkNorm > normG * patMin(kfgr,pow(normG,theta))
	 && iter <= maxIter) {
    
    ++iter;
    //    DEBUG_MESSAGE("******** CG ITER " << iter) ;
    trVector Hp = (*H)(pk,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trUNKNOWN ;
    }
    
    patReal kappak = inner_product(pk.begin(),pk.end(),Hp.begin(),0.0) ;
    if (kappak <= 0) {
      // Negative curvature found
      patReal sigma = (- skMpk + 
		       sqrt(skMpk*skMpk + 
			    normMpk2 * (radius*radius - normMsk2))) 
	/ normMpk2 ;
      solution = solution + sigma * pk ;
      normMsk2 += 2.0 * sigma * skMpk + sigma * sigma * normMpk2 ;
      
      status = trNEG_CURV ;
      //      DEBUG_MESSAGE("NEG_CURV") ;
      return status ;
    }
    
    patReal alphak = gkvk / kappak ;
    
    patReal normPrec = 
      normMsk2 + 2.0 * alphak * skMpk + alphak * alphak * normMpk2 ;

    if (sqrt(normPrec) >= radius) {
      // The iterations leave the trust region
      patReal sigma = (- skMpk + 
		       sqrt(skMpk*skMpk + 
			    normMpk2 * (radius*radius - normMsk2))) 
	/ normMpk2 ;
      solution = solution + sigma * pk ;
      normMsk2 += 2.0 * sigma * skMpk + sigma * sigma * normMpk2 ;
      //      DEBUG_MESSAGE("OUT_OF_TR") ;
      status =  trOUT_OF_TR;
      return status ;
    }
    
    solution += alphak * pk ;
    gk += alphak * Hp ;
    gkNorm = sqrt(inner_product(gk.begin(),gk.end(),gk.begin(),0.0)) ;
    vk = M->solve(&gk,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trUNKNOWN ;
    }
    patReal newGkvk = inner_product(gk.begin(),gk.end(),vk.begin(),0.0) ;
    patReal betak = newGkvk / gkvk ;
    gkvk = newGkvk ;
    pk = -vk + betak * pk ;

   normMsk2 += 2.0 * alphak * skMpk + alphak * alphak * normMpk2 ;

   skMpk = betak * (skMpk + alphak * normMpk2) ;
   normMpk2 = gkvk + betak * betak * normMpk2 ;

  }

  if (iter > maxIter) {
    //   DEBUG_MESSAGE("MAXITER") ;
    return trMAXITER ;
  }
  else {
    //    DEBUG_MESSAGE("CONV") ;
    return trCONV ;
  }
  return trUNKNOWN ;
}

trVector trTointSteihaug::getSolution(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }
  return solution ;

}

patReal trTointSteihaug::getValue(patError*& err) {
  // This function returns m(x+s)-m(x), that is gTs + 0.5 sTHs

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  trVector hs = (*H)(solution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  return (inner_product(g->begin(),g->end(),solution.begin(),0.0) +
	  0.5 * inner_product(hs.begin(),hs.end(),solution.begin(),0.0)) ; 
}

patReal trTointSteihaug::getNormStep() {
  return sqrt(normMsk2) ;
}

patString trTointSteihaug::getStatusName(trTermStatus status) {
  return statusName[status] ;
}
