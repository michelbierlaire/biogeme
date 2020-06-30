//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLinearUtility.cc
// @date   Wed Jul 10 08:10:53 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#include <sstream>
#include <cmath>
#include "bioSmartPointer.h"
#include <vector>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprLinearUtility.h"

bioExprLinearUtility::bioExprLinearUtility(std::vector<bioLinearTerm> t):
  listOfTerms(t) {
  for (std::vector<bioLinearTerm>::iterator i = listOfTerms.begin() ;
       i != listOfTerms.end() ;
       ++i) {
    bioBoolean problem = false ;
    if (i->theBeta == NULL) {
      DEBUG_MESSAGE("Null pointer to beta") ;
      problem = true ;
    }
    if (i->theVar == NULL) {
      DEBUG_MESSAGE("Null pointer to var") ;
      problem = true ;
    }
    if (problem) {
      throw bioExceptions(__FILE__,__LINE__,"Null pointer in linear utility expression") ;
    }
      
    
    listOfChildren.push_back(i->theBeta) ;
    listOfChildren.push_back(i->theVar) ;
    theFriend[i->theBetaId] = i->theVarName ;
    theFriend[i->theVarId] = i->theBetaName ;
  }
}

bioExprLinearUtility::~bioExprLinearUtility() {
}

bioSmartPointer<bioDerivatives>
bioExprLinearUtility::getValueAndDerivatives(std::vector<bioUInt> literalIds,
					     bioBoolean gradient,
					     bioBoolean hessian) {

  if (!gradient && hessian) {
    throw bioExceptions(__FILE__,__LINE__,"If the hessian is needed, the gradient must be computed") ;
  }
  
  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;

  bioUInt n = literalIds.size() ;
  theDerivatives->f = 0.0 ;
  std::map<bioString,bioReal> values = getAllLiteralValues() ;
  for (std::vector<bioLinearTerm>::iterator i =
	 listOfTerms.begin() ;
       i != listOfTerms.end() ;
       ++i) {
    if ((values[i->theBetaName] != 0.0) &&  (values[i->theVarName] != 0.0)) {
      theDerivatives->f += values[i->theBetaName] * values[i->theVarName] ;
      //      DEBUG_MESSAGE(i->theBetaName << " * " <<  i->theVarName << " = " << values[i->theBetaName] * values[i->theVarName]) ;
    }
  }
  if (gradient) {
    if (hessian) {
      theDerivatives->setDerivativesToZero() ;
    }
    else {
      theDerivatives->setGradientToZero() ;
    }
    for (std::size_t i = 0 ; i < literalIds.size() ; ++i) {
      theDerivatives->g[i] = values[theFriend[i]] ;
      //      DEBUG_MESSAGE("Gradient[" << i << "]=" << "value[" << theFriend[i] << "]=" <<theDerivatives->g[i]) ;
      
    }
  } 
  return theDerivatives ;
}

bioString bioExprLinearUtility::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "bioLinearUtility[" ;
  for (std::vector<bioLinearTerm>::const_iterator i =
	 listOfTerms.begin() ;
       i != listOfTerms.end() ;
       ++i) {
    if (i != listOfTerms.begin()) {
      str << " + " ;
    }
    str << i->theBeta->print(hp) << " * " << i->theVar->print(hp) ;
  }
  str << "]" ;
  return str.str() ;
}
