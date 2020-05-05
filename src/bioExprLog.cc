//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLog.cc
// @date   Tue Apr 17 12:16:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprLog.h"
#include "bioDebug.h"
#include "bioExceptions.h"
#include <cmath>
#include <sstream>

bioExprLog::bioExprLog(bioExpression* c) :
  child(c) {
  listOfChildren.push_back(c) ;
}

bioExprLog::~bioExprLog() {
  
}

bioDerivatives* bioExprLog::getValueAndDerivatives(std::vector<bioUInt> literalIds,
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


  bioUInt n = literalIds.size() ;
  bioDerivatives* childResult = child->getValueAndDerivatives(literalIds,gradient,hessian) ;
  if (childResult->f < 0) {
    if (std::abs(childResult->f) < 1.0e-6) {
      childResult->f = 0.0 ;
    }
    else {
      std::stringstream str ;
      str << "Current values of the literals: " << std::endl ;
      std::map<bioString,bioReal> m = getAllLiteralValues() ;
      for (std::map<bioString,bioReal>::iterator i = m.begin() ;
	   i != m.end() ;
	   ++i) {
	str << i->first << " = " << i->second << std::endl ;
      }
      if (rowIndex != NULL) {
	str << "row number: " << *rowIndex << ", ";
      }
      
      str << "Cannot take the log of a non positive number [" << childResult->f << "]" << std::endl ;
      throw bioExceptions(__FILE__,__LINE__,str.str()) ;
    }
  }
  if (childResult->f == 0.0) {
    theDerivatives->f = -std::numeric_limits<bioReal>::max() / 2.0 ;
  }
  else {    
    theDerivatives->f = log(childResult->f) ;
  }
  if (gradient) {
    for (bioUInt i = 0 ; i < n ; ++i) {
      theDerivatives->g[i] = childResult->g[i] / childResult->f ;
      if (hessian) {
	for (bioUInt j = 0 ; j < n ; ++j) {
	  bioReal fsquare = childResult->f * childResult->f ;
	  theDerivatives->h[i][j] = childResult->h[i][j] / childResult->f -  childResult->g[i] *  childResult->g[j] / fsquare ;
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprLog::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "log(" << child->print(hp) << ")";
  return str.str() ;
}

