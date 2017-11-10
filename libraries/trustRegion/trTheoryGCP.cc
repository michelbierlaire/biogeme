//-*-c++-*------------------------------------------------------------
//
// File name : trTheoryGCP.cc
// Date :      Mon Nov 27 09:33:43 2000
//
//--------------------------------------------------------------------

#include <numeric>
#include "patMath.h"
#include "trTheoryGCP.h"
#include "patDisplay.h"

trTheoryGCP::trTheoryGCP(const trBounds& _bounds,
			 patReal _radius,
			 const trVector& _currentIterate,
			 const trVector& _g,
			 trMatrixVector& _H,
			 trParameters p) :
  bounds(_bounds),
  radius(_radius),
  currentIterate(_currentIterate),
  g(_g),
  H(_H),
  tmax(patMaxReal),
  tmin(0.0),
  theParameters(p){

  WARNING("This routine has not been tested... yet") ;

  t = radius / sqrt(inner_product(g.begin(),g.end(),g.begin(),0.0)) ;

}

patBoolean trTheoryGCP::checkConditions1221(patError*& err) {

  
  // Conditions (12.2.1) p. 453

  if (normSk > radius) {
    return patFALSE ;
  }

  // Here, mk is defined as m(projection)-m(currentIterate) = s_k^T g_k +
  // s_k^T H s_k. Note that, in the book, mk is the value of the quadratic
  // model m(projection).

  if ( mk > theParameters.kappaUbs * gksk) {
    return patFALSE ;
  }

  return patTRUE ;
}

patBoolean trTheoryGCP::checkConditions1222(patError*& err) {

  
  // Conditions 12.2.2 p. 453

  if (normSk >=  theParameters.kappaFrd * radius) {
    return patTRUE ;
  }

  if (mk >=  theParameters.kappaLbs * gksk) {
    return patTRUE ;
  }

  // The projection onto the tangent cone is obtained, in the case of simple
  // bounds constrints, by projecting -g onto the feasible set itself.

  trVector gProj = bounds.getProjection(projection - g,err) - projection ;
  
  patReal gProjNorm = 
    sqrt(inner_product(gProj.begin(),gProj.end(),gProj.begin(),0.0)) ;

  if (gProjNorm 
      <= theParameters.kappaEpp * patAbs(gksk) / radius) {
    return patTRUE ;
  }

  return patFALSE ;
  
}

void trTheoryGCP::computeProjections(patError*& err) {
  
  /**
     projection is p(t,x_k)
   */
  
  projection = bounds.getProjection(currentIterate - t * g,err) ;
  sk = projection - currentIterate ;

  trVector Hs = H(sk,err) ;
  if (err != NULL) {
    WARNING(err->describe()) ;
    return ;
  }
  mk = inner_product(g.begin(),g.end(),sk.begin(),0.0) +
    0.5 * inner_product(sk.begin(),sk.end(),Hs.begin(),0.0) ;

  normSk = sqrt(inner_product(sk.begin(),sk.end(),sk.begin(),0.0)) ;
  gksk = inner_product(sk.begin(),sk.end(),g.begin(),0.0) ;

}


trVector trTheoryGCP::computeGCP(patError*& err) {
  do {
    computeProjections(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trVector();
    }

    patBoolean c1221 = checkConditions1221(err) ;
    if (err != NULL) {
      WARNING(err->describe()) ;
      return trVector();
    }
    if (!c1221) {
      tmax = t ;
    }
    else {
      patBoolean c1222 = checkConditions1222(err) ;
      if (err != NULL) {
	WARNING(err->describe()) ;
	return trVector();
      }
      if (!c1222) {
	tmin = t ;
      }
      else {
	return projection ;
      }

      if (tmax == patMaxReal) {
	t *= 2.0 ;
      }
      else {
	t = 0.5 * (tmin + tmax) ;
      }
    }
  } while (patTRUE) ;
}
