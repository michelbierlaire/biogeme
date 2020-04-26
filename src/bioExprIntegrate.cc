//-*-c++-*------------------------------------------------------------
//
// File name : bioExprIntegrate.cc
// @date   Wed May  9 17:38:29 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprIntegrate.h"
#include <sstream>
#include <cmath>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprGaussHermite.h"


bioExprIntegrate::bioExprIntegrate(bioExpression* c, bioUInt id) :
  child(c), rvId(id) {
  listOfChildren.push_back(c) ;
}
bioExprIntegrate::~bioExprIntegrate() {

}

bioDerivatives* bioExprIntegrate::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {

  if (theDerivatives == NULL) {
    theDerivatives = new bioDerivatives(literalIds.size()) ;
  }

  bioExprGaussHermite theGh(child,literalIds,rvId,gradient,hessian) ;   
  bioGaussHermite theGhAlgo(&theGh) ;
  std::vector<bioReal> r = theGhAlgo.integrate() ;
  theDerivatives->f = r[0] ;
  bioUInt n = literalIds.size() ;
  if (gradient) {
    for (bioUInt j = 0 ; j < n ; ++j) {
      if (std::isfinite(r[j+1])) {
	theDerivatives->g[j] = r[j+1] ;
      }
      else {
	theDerivatives->g[j] = bioMaxReal ;
      }
    }
  }
  if (hessian) {
    bioUInt index = 1 + theDerivatives->g.size() ;
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	if (std::isfinite(r[index])) {
	  theDerivatives->h[i][j] = theDerivatives->h[j][i] = r[index] ;
	}
	else {
	  theDerivatives->h[i][j] = theDerivatives->h[j][i] = bioMaxReal ;
	}
	++index ;
      }
    }
  }

  return theDerivatives ;
}

bioString bioExprIntegrate::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "Integrate(" << child->print(hp) << "," << rvId << ")" ;
  return str.str() ;

}
