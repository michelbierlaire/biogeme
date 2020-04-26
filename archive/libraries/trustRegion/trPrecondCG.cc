//-*-c++-*------------------------------------------------------------
//
// File name : trPrecondCG.cc
// Author :    Michel Bierlaire
// Date :      Mon Nov 27 16:59:42 2000
//
// Implementation of algorithm 5.1.4: the preconditioned CG metohd
//
// Source: Conn, Gould Toint (2000) Trust Region Methods
//--------------------------------------------------------------------

#include <numeric>
#include <cmath>
#include "patDisplay.h"
#include "patMath.h"
#include "patErrNullPointer.h"
#include "trPrecondCG.h"
#include "trVector.h"
#include "trMatrixVector.h"
#include "trPrecond.h"

#include "trHessian.h"

trPrecondCG::trPrecondCG(const  trVector& _g,
			 trMatrixVector* _H,
			 const trBounds& _bounds,
			 const trPrecond* _m,
			 trParameters p,
			 patError*& err) :
  g(_g),
  H(_H),
  M(_m),
  bounds(_bounds),
  status(trUNKNOWN),
  theParameters(p),
  solution(g.size()),
  statusName(5){

  if (_H == NULL) {
    err = new patErrNullPointer("trMatrixVector") ;
    WARNING(err->describe()) ;
    return ;
    
  }
  if (_m == NULL) {
    err = new patErrNullPointer("trPrecond") ;
    WARNING(err->describe()) ;
    return ;
    
  }


  normG = sqrt(inner_product(g.begin(),g.end(),g.begin(),0.0)) ;
  maxIter = 5*g.size() ;
  
  statusName[trUNKNOWN] =   "Unknown " ;
  statusName[trNEG_CURV] =  "NegCurv " ;
  statusName[trOUT_OF_TR] = "OutTrReg" ;
  statusName[trMAXITER] =   "MaxIter " ;
  statusName[trCONV] =      "Converg " ;

}
				   
trPrecondCG::trTermStatus trPrecondCG::getTermStatus(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }
  return status ;
}

trPrecondCG::trTermStatus trPrecondCG::run(patError*& err) {

//   DEBUG_MESSAGE("Start CG") ;

  if (err != NULL) {
    WARNING(err->describe()) ;
    return trUNKNOWN ;
  }

  fill(solution.begin(),solution.end(),0.0) ;

  trVector gk(g) ;
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

//    DEBUG_MESSAGE("gk=" << gk) ;
//    DEBUG_MESSAGE("vk=" << vk) ;

  trVector pk = -vk ;

  patULong iter = 0 ;

  patReal gkvk = inner_product(gk.begin(),gk.end(),vk.begin(),0.0) ;

  patULong unfeasibleIter = 0 ;

  while (gkNorm > normG * patMin(theParameters.fractionGradientRequired,pow(normG,theParameters.expTheta))
	 && iter <= maxIter) {
    
    ++iter;

//     DEBUG_MESSAGE("===== iter " << iter << "========") ;

    if (H == NULL) {
      err = new patErrNullPointer("trMatrixVector") ;
      WARNING(err->describe()) ;
      return trUNKNOWN ;
    }
    trVector Hp = (*H)(pk,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trUNKNOWN ;
    }
    
    patReal kappak = inner_product(pk.begin(),pk.end(),Hp.begin(),0.0) ;

    if (kappak <= 0) {
      // Negative curvature found
      bounds.getMaxStep(solution,pk,&solution,err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return trUNKNOWN ;
      }
      status = trNEG_CURV ;
      return status ;
    }
    
    patReal alphak = gkvk / kappak ;

    trVector candidate = solution + alphak * pk ;

    patBoolean ok = bounds.isFeasible(candidate,err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trUNKNOWN ;
    }

    if (!ok) {
      ++unfeasibleIter ;

      if (unfeasibleIter >= theParameters.infeasibleCgIter) {
	// 	DEBUG_MESSAGE("Next iterate out of feasible region") ;
	// Next iterate out of the feasible region
	trVector newPoint(solution.size())  ;
	
	bounds.getMaxStep(solution,pk,&newPoint,err) ;
	//       DEBUG_MESSAGE("Max step =" << alpha) ;
	//       DEBUG_MESSAGE("New point = " << newPoint) ;
	if (err != NULL) {
	  WARNING(err->describe()) ;
	  return trUNKNOWN ;
	}
	solution = newPoint ;
	status = trOUT_OF_TR ;
	return status ;
      }
    }
    else {
      unfeasibleIter = 0 ;
    }

    //DEBUG_MESSAGE("Unfeasible CG iter: " << unfeasibleIter) ;
    solution = candidate ;
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

    //    DEBUG_MESSAGE("gknorm = " << gkNorm) ;

  }

  if (iter > maxIter) {
    return trMAXITER ;
  }
  else {
    return trCONV ;
  }
  return trUNKNOWN ;
}

trVector trPrecondCG::getSolution(patError*& err) {
  if (err != NULL) {
    WARNING(err->describe()) ;
    return trVector() ;
  }
  return solution ;

}

patReal trPrecondCG::getValue(patError*& err) {
  // This function returns m(x+s)-m(x), that is gTs + 0.5 sTHs

  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }
  
  if (H == NULL) {
    err = new patErrNullPointer("trMatrixVector") ;
    WARNING(err->describe()) ;
    return patReal() ;
  }

  trVector hs = (*H)(solution,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return patReal() ;
  }

  patReal gTs = inner_product(g.begin(),g.end(),solution.begin(),0.0) ;
  patReal sHs = inner_product(hs.begin(),hs.end(),solution.begin(),0.0) ;
  
  return (gTs + 0.5 * sHs) ; 
}

patString trPrecondCG::getStatusName(trTermStatus status) {
  return statusName[status] ;
}
