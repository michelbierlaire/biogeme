//-*-c++-*------------------------------------------------------------
//
// File name : bioSeveralExpressions.cc
// @date   Wed Mar  3 18:08:39 2021
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioSeveralExpressions.h"
#include "bioExceptions.h"
#include "bioDebug.h"
#include <sstream>

bioSeveralExpressions::bioSeveralExpressions(std::vector<bioExpression*> exprs):
  theExpressions(exprs) {

  for (std::vector<bioExpression*>::iterator i = exprs.begin() ;
       i != exprs.end();
       ++i) {
    listOfChildren.push_back(*i) ;
  }
}

bioSeveralExpressions::~bioSeveralExpressions() {

}

bioString bioSeveralExpressions::print(bioBoolean hp) const {

  std::stringstream str ;
  str << "MultipleExpressions[" ;
  for (std::vector<bioExpression*>::const_iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    if ((*i) == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"bioExpression") ;
    }
    if (i != theExpressions.begin()) {
      str << ", " ;
    }
    str << (*i)->print(hp) ;
  }
  str << "]" ;
  return str.str() ;
}

bioReal bioSeveralExpressions::getValue() {
  std::stringstream str ;
  str << "This function cannot be called for multiple expressions." ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
}

std::vector<bioReal > bioSeveralExpressions::getValues() {
  std::vector<bioReal > results ;
  for (std::vector<bioExpression*>::iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    bioReal res ;
    try {
      res = (*i)->getValue() ;
    }
    //    catch(bioExceptions& e) {
    catch(...) {
      res = std::numeric_limits<bioReal>::quiet_NaN() ;
    }
    results.push_back(res) ;
  }
  return results ;
}

const bioDerivatives* bioSeveralExpressions::getValueAndDerivatives(std::vector<bioUInt> literalIds,
								    bioBoolean gradient,
								    bioBoolean hessian) {
  std::stringstream str ;
  str << "This function cannot be called for multiple expressions." ;
  throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  
}

std::vector<const bioDerivatives*>
bioSeveralExpressions::getAllValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) {
  std::vector<const bioDerivatives* > results ;
  for (std::vector<bioExpression*>::iterator i = theExpressions.begin() ;
       i != theExpressions.end() ;
       ++i) {
    results.push_back((*i)->getValueAndDerivatives(literalIds, gradient, hessian)) ;
  }
  return results ;
}
