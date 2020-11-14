//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNumeric.cc
// @date   Fri Apr 13 15:12:00 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprNumeric.h"
#include <sstream>
#include "bioDebug.h"
bioExprNumeric::bioExprNumeric(bioReal v) : value(v) {
  
}

bioExprNumeric::~bioExprNumeric() {

}

const bioDerivatives* bioExprNumeric::getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) {

  if (gradient && theDerivatives.getSize() != literalIds.size()) {
    theDerivatives.resize(literalIds.size()) ;
  }

  if (gradient) {
    if (hessian) {
      theDerivatives.setDerivativesToZero() ;
    }
    else {
      theDerivatives.setGradientToZero() ;
    }
  }
  theDerivatives.f = value ;
  
  return &theDerivatives ;
}

bioString bioExprNumeric::print(bioBoolean hp) const {
  std::stringstream str ;
  str << value ;
  return str.str() ;
}

