//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPanelTrajectory.cc
// @date   Mon May 21 13:48:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprPanelTrajectory.h"
#include <cmath>
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprPanelTrajectory::bioExprPanelTrajectory(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprPanelTrajectory::~bioExprPanelTrajectory() {

}

bioDerivatives* bioExprPanelTrajectory::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							  bioBoolean gradient,
							  bioBoolean hessian) {


  if (theDerivatives == NULL) {
    theDerivatives = new bioDerivatives(literalIds.size()) ;
  }
  else {
    if (gradient && theDerivatives->getSize() != literalIds.size()) {
      delete(theDerivatives) ;
      theDerivatives = new bioDerivatives(literalIds.size()) ;
    }
  }

  theDerivatives->f = 0.0 ;
  if (gradient) {
    if (hessian) {
      theDerivatives->setDerivativesToZero() ;
    }
    else {
      theDerivatives->setGradientToZero() ;
    }
  }

  if (dataMap == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"data map") ;
  }

  if (individualIndex == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"individual index") ;

  }
  
  if (*individualIndex >= dataMap->size()) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,*individualIndex,0,dataMap->size() - 1) ;
  }
  bioUInt n = literalIds.size() ;
  child->setRowIndex(&theRowIndex) ;

  for (theRowIndex = (*dataMap)[*individualIndex][0]  ; theRowIndex <= (*dataMap)[*individualIndex][1] ; ++theRowIndex) {
    bioDerivatives* childResult(NULL) ;
    try {
      childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
      // if (childResult->f <= 1.0e-6) {
      // 	std::stringstream str ;
      // 	str << "Error for data entry " << theRowIndex << ": probability " << childResult->f << "for " << child->print() ;
      // 	throw bioExceptions(__FILE__,__LINE__,str.str()) ;
      // }
      theDerivatives->f += log(childResult->f) ;
      if (gradient) {
	for (bioUInt i = 0 ; i < n ; ++i) {
	  if (childResult->g[i] != 0.0) {
	    theDerivatives->g[i] += childResult->g[i] / childResult->f ;
	  }
	  if (hessian) {
	    for (bioUInt j = i ; j < n ; ++j) {
	      if (childResult->h[i][j] != 0.0) {
		theDerivatives->h[i][j] += childResult->h[i][j] / childResult->f ;
	      }
	      if (childResult->g[i] != 0.0 && childResult->g[j] != 0.0) {
		theDerivatives->h[i][j] -= childResult->g[i] * childResult->g[j] / (childResult->f * childResult->f) ;
	      }
	    }
	  }
	}
      }
    }
    catch(bioExceptions& e) {
      std::stringstream str ;
      str << "Error for data entry " << theRowIndex << ": " << e.what() ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  // So far, we have calculated the derivatrives for the log
  // likelihood. We need now to store the derivatives of the
  // likelihood of the trajectory.

  theDerivatives->f = exp(theDerivatives->f) ;
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      if (hessian) {
	for (bioUInt j = i ; j < n ; ++j) {
	  if (theDerivatives->g[i] != 0.0 && theDerivatives->g[j] != 0.0) {
	    theDerivatives->h[i][j] += theDerivatives->g[i] * theDerivatives->g[j] ;
	  }
	  theDerivatives->h[i][j] *= theDerivatives->f ;
	}
      }
      theDerivatives->g[i] *= theDerivatives->f ;
    }
  }
  if (hessian) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      for (bioUInt j = i ; j < n ; ++j) {
	theDerivatives->h[j][i] = theDerivatives->h[i][j] ;
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprPanelTrajectory::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "PanelLikelihoodTrajectory(" << child->print(hp) << ")";
  return str.str() ;

}

// This is the only class that should not propagate the row index
// pointer, as its own pointer may be inconsistent with its formula.
void bioExprPanelTrajectory::setRowIndex(bioUInt* i) {
  
}
