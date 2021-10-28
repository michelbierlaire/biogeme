//-*-c++-*------------------------------------------------------------
//
// File name : bioExprElem.cc
// @date   Wed Apr 18 10:32:21 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include <sstream>
#include <cmath>
#include "bioDebug.h"
#include "bioExceptions.h"
#include "bioExprElem.h"

bioExprElem::bioExprElem(bioExpression* k, std::map<bioUInt,bioExpression*> d):
  key(k), dictOfExpressions(d) {


  listOfChildren.push_back(k) ;
  for (std::map<bioUInt,bioExpression*>::iterator i = d.begin() ;
       i != d.end();
       ++i) {
    if (i->second == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"Null expression in dictionary") ;
    }
    listOfChildren.push_back(i->second) ;
  }

}

bioExprElem::~bioExprElem() {

}

const bioDerivatives* bioExprElem::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						    bioBoolean gradient,
				      bioBoolean hessian) {

  theDerivatives.with_g = gradient ;
  theDerivatives.with_h = hessian ;

  theDerivatives.resize(literalIds.size()) ;


  bioUInt k = bioUInt(key->getValue()) ;

  std::map<bioUInt,bioExpression*>::const_iterator found = dictOfExpressions.find(k) ;
  if (found == dictOfExpressions.end()) {
    std::stringstream str ;
    str << "Key (" << key->print(true) << "=" << k << ") is not present in dictionary: " << std::endl;
    for (std::map<bioUInt,bioExpression*>::const_iterator i = dictOfExpressions.begin() ;
	 i != dictOfExpressions.end() ;
	 ++i) {
      str << "  " << i->first << ": " << i->second->print(true) << std::endl ;
    }
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  if (found->second == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"expression") ;
  }
  const bioDerivatives* fgh =  found->second->getValueAndDerivatives(literalIds,gradient,hessian) ;
  if (fgh == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"derivatives") ;
  }
  theDerivatives.f = fgh->f ;
  if (!std::isfinite(fgh->f)) {
    std::stringstream str ;
    str << "Invalid value for expression <" << found->second->print(true) << ">: " << fgh->f ;
    throw bioExceptions(__FILE__,__LINE__,str.str()) ;
  }
  if (gradient) {
    for (std::size_t k = 0 ; k < literalIds.size() ; ++k) {
      theDerivatives.g[k] = fgh->g[k] ;
      if (hessian) {
	for (std::size_t l = 0 ; l < literalIds.size() ; ++l) {
	  theDerivatives.h[k][l] = fgh->h[k][l] ;
	}
      }
    }
  }
  return &theDerivatives ;
}

bioString bioExprElem::print(bioBoolean hp) const {
  std::stringstream str ;
  str << "Elem[" << key->print(hp) << "](" ;
  for (std::map<bioUInt,bioExpression*>::const_iterator i = dictOfExpressions.begin() ;
       i != dictOfExpressions.end() ;
       ++i) {
    if (i != dictOfExpressions.begin()) {
      str << ";" ;
    }
    str << i->first << ":" << i->second->print(hp) ;
  }
  str << ")" ;
  return str.str() ;
}
