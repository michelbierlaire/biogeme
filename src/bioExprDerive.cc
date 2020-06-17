//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDerive.cc
// @date   Tue May  1 21:03:46 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprDerive.h"
#include <sstream>
#include "bioSmartPointer.h"
#include "bioDebug.h"
#include "bioExceptions.h"


bioExprDerive::bioExprDerive(bioSmartPointer<bioExpression>  c, bioUInt lid) :
  child(c), literalId(lid) {
  listOfChildren.push_back(c) ;
}
bioExprDerive::~bioExprDerive() {
}

bioSmartPointer<bioDerivatives> bioExprDerive::getValueAndDerivatives(std::vector<bioUInt> literalIds,
								      bioBoolean gradient,
								      bioBoolean hessian) {

  theDerivatives = bioSmartPointer<bioDerivatives>(new bioDerivatives(literalIds.size())) ;
  if (gradient || hessian) {
    throw bioExceptions(__FILE__,__LINE__,"No derivatives are available for this expression, yet.") ;
  }
  
  std::vector<bioUInt> theIds ;
  theIds.push_back(literalId) ;

  bioSmartPointer<bioDerivatives> childResult = child->getValueAndDerivatives(theIds,true,false) ;
  if (childResult == NULL) {
    throw bioExceptNullPointer(__FILE__,__LINE__,"derivatives") ;
  }
  theDerivatives->f = childResult->g[0] ;
  return theDerivatives ;
}

bioString bioExprDerive::print(bioBoolean hp) const {
  std::stringstream str ; 
  str << "Derive(" << child->print(hp) << "," << literalId << ")" ;
  return str.str() ;

}
