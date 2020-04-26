//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDraws.cc
// @date   Mon May  7 10:24:27 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#include "bioExprDraws.h"
#include <sstream>
#include "bioExceptions.h"
#include "bioDebug.h"

bioExprDraws::bioExprDraws(bioUInt literalId, bioUInt drawId, bioString name) : bioExprLiteral(literalId,name), theDrawId(drawId), drawIndex(NULL) {
  
}
bioExprDraws::~bioExprDraws() {
}


bioString bioExprDraws::print(bioBoolean hp) const {
  std::stringstream str ;
  str << theName << " lit[" << theLiteralId << "],draw[" << theDrawId << "]" ;
  return str.str() ;
}

void bioExprDraws::setDrawIndex(bioUInt* d) {
  drawIndex = d ;
}

bioReal bioExprDraws::getLiteralValue() const {
  if (draws == NULL) {
      throw bioExceptNullPointer(__FILE__,__LINE__,"draws") ;
  }
  if (sampleSize == 0) {
    throw bioExceptions(__FILE__,__LINE__,"Empty list of draws.") ;
  }
  if (numberOfDraws == 0) {
    throw bioExceptions(__FILE__,__LINE__,"Empty list of draws.") ;

  }
  if (numberOfDrawVariables == 0) {
    throw bioExceptions(__FILE__,__LINE__,"Empty list of draws.") ;
  }
  if (individualIndex == NULL) {
    throw bioExceptions(__FILE__,__LINE__,"Row index is not defined.") ;
  }
  if (*individualIndex >= sampleSize) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,*individualIndex,0,sampleSize-1) ;
  }
  if (drawIndex == NULL) {
    throw bioExceptions(__FILE__,__LINE__,"Draw index is not defined. It may be caused by the use of draws outside a Montecarlo statement.") ;
  }
  if (*drawIndex >= numberOfDraws) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,*drawIndex,0,numberOfDraws-1) ;
  }
  if (theDrawId == bioBadId || theDrawId >= numberOfDrawVariables) {
    throw bioExceptOutOfRange<bioUInt>(__FILE__,__LINE__,theDrawId,0,numberOfDrawVariables-1) ;
  }

  return (*draws)[*individualIndex][*drawIndex][theDrawId] ;

}

