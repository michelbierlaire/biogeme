//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLogLogitFullChoiceSet.cc
// @date   Fri Apr 13 15:16:24 2018
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <sstream>
#include <cmath>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprLogLogitFullChoiceSet.h"

bioExprLogLogitFullChoiceSet::bioExprLogLogitFullChoiceSet(bioExpression* c,
				 std::map<bioUInt,bioExpression*> u) :
  choice(c), utilities(u) {
  listOfChildren.push_back(choice) ;
  for (std::map<bioUInt,bioExpression*>::iterator i = u.begin() ;
       i != u.end();
       ++i) {
    listOfChildren.push_back(i->second) ;
  }
}

bioExprLogLogitFullChoiceSet::~bioExprLogLogitFullChoiceSet() {

}

bioDerivatives* bioExprLogLogitFullChoiceSet::getValueAndDerivatives(std::vector<bioUInt> literalIds,
							bioBoolean gradient,
							bioBoolean hessian) {


  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }
  
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
  bioUInt chosen = bioUInt(choice->getValue()) ;
  std::vector<bioDerivatives*> Vs ;
  bioDerivatives* chosenUtility(NULL) ;
  bioDerivatives* V;
  bioReal largestUtility(-bioMaxReal) ;
  for (std::map<bioUInt,bioExpression*>::iterator theUtil = utilities.begin() ;
       theUtil != utilities.end() ;
       ++theUtil) {
      V = theUtil->second->getValueAndDerivatives(literalIds,gradient,hessian) ;
      if (V->f > largestUtility) {
	largestUtility = V->f ;
      }
      if (theUtil->first == chosen) {
	chosenUtility = V ;
      }
      Vs.push_back(V) ;
    }

  if (chosenUtility == NULL) {
    std::stringstream str ;
    str << "Alternative "
	<< chosen
	<< " is not known. The alternatives that have been defined are" ;
    for (std::map<bioUInt,bioExpression*>::iterator i = utilities.begin() ;
	 i != utilities.end() ;
	 ++i) {
      str << " " << i->first ;
    }
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  bioReal maxexp = ceil(largestUtility / 10.0) * 10.0 ;

  for (std::vector<bioDerivatives*>::iterator i = Vs.begin() ;
       i != Vs.end() ;
       ++i) {
    (*i)->f -= maxexp ;
  }

  std::vector<bioReal> expi(Vs.size()) ; ;
    
  bioReal denominator(0.0) ;
  for (bioUInt k = 0 ; k < Vs.size() ; ++k) {
    expi[k] = exp(Vs[k]->f) ;
    denominator += expi[k] ;
  }

  theDerivatives->f = chosenUtility->f - log(denominator) ;
  if (gradient) {
    std::vector<bioReal> weightedSum(n,0.0) ;
    for (bioUInt j = 0 ; j < n ; ++j) {
      for (bioUInt k = 0 ; k < Vs.size() ; ++k) {
	if (Vs[k]->g[j] != 0.0) {
	  weightedSum[j] += Vs[k]->g[j] * expi[k] ;
	}
      }
      theDerivatives->g[j] = chosenUtility->g[j] ;
      if (weightedSum[j] != 0.0) {
	theDerivatives->g[j] -= weightedSum[j] / denominator ;
      }
    }
    
    if (hessian) {
      bioReal dsquare = denominator * denominator ;
      for (bioUInt i = 0 ; i < n ; ++i) {
	for (bioUInt j = i ; j < n ; ++j) {
	  bioReal dsecond(0.0) ;
	  for (bioUInt k = 0 ; k < Vs.size() ; ++k ) {
	    if (Vs[k]->g[i] != 0 && Vs[k]->g[j] != 0.0) {
	      dsecond += expi[k] * Vs[k]->g[i] * Vs[k]->g[j] ;
	    }
	    bioReal vih = Vs[k]->h[i][j] ;
	    if (vih != 0.0) {
	      dsecond += expi[k] * vih ;
	    }
	  }
	  bioReal v =  chosenUtility->h[i][j] ;
	  bioReal v1(0.0) ;
	  if (weightedSum[i] != 0.0 && weightedSum[j] != 0.0) {
	    v1 = weightedSum[i] * weightedSum[j] / dsquare ;
	  }
	  bioReal v2 = dsecond / denominator ;
	  theDerivatives->h[i][j] = theDerivatives->h[j][i] = v+v1-v2 ;
	}
      }
    }
  }
  return theDerivatives ;
}

bioString bioExprLogLogitFullChoiceSet::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "LogitFullChoiceSet[" << choice->print(hp) << "](" ;
  for (std::map<bioUInt,bioExpression*>::const_iterator i = utilities.begin() ;
       i != utilities.end() ;
       ++i) {
    if (i != utilities.begin()) {
      str << ";" ;
    }
    str << i->second->print(hp) ;
  }
  str << ")" ;
  return str.str() ;
}
