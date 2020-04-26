//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMatrix.cc
// Author :    Michel Bierlaire
// Date :      Mon Oct  1 09:50:12 2012
//
//--------------------------------------------------------------------

#include "bioArithMatrix.h"
#include "patErrOutOfRange.h"
#include "bioExpression.h"
#include "bioExpressionRepository.h"

bioArithMatrix::bioArithMatrix(patULong dim, bioExpressionRepository* theRep) : 
  theMatrix(dim,vector< patULong >(dim)),
  theRepository(theRep) {

}

void bioArithMatrix::setExpression(bioExpression* theExpr, 
				   patULong row, 
				   patULong col, 
				   patError*& err) {
  
  if (row >= getSize()) {
    err = new patErrOutOfRange<patULong>(row,0,getSize()-1) ;
    WARNING(err->describe()) ;
    return ;
  }
  if (col >= getSize()) {
    err = new patErrOutOfRange<patULong>(row,0,getSize()-1) ;
    WARNING(err->describe()) ;
    return ;
  }
  theMatrix[row][col] = theExpr->getId() ;

}
bioExpression* bioArithMatrix::getExpression(patULong row, 
					     patULong col, 
					     patError*& err) {
  if (row >= getSize()) {
    err = new patErrOutOfRange<patULong>(row,0,getSize()-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  if (col >= getSize()) {
    err = new patErrOutOfRange<patULong>(row,0,getSize()-1) ;
    WARNING(err->describe()) ;
    return NULL ;
  }
  return theRepository->getExpression(theMatrix[row][col]) ;
}

patULong bioArithMatrix::getSize() const {
  return theMatrix.size() ;
}
