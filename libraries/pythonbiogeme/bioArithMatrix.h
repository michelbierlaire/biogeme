//-*-c++-*------------------------------------------------------------
//
// File name : bioArithMatrix.h
// Author :    Michel Bierlaire
// Date :      Mon Oct  1 09:38:37 2012
//
//--------------------------------------------------------------------

#ifndef bioArithMatrix_h
#define bioArithMatrix_h

#include "patError.h"
#include "patType.h"

class bioExpression ;
class bioExpressionRepository ;

// Implements a square matrix of expressions. 

class bioArithMatrix {

  bioArithMatrix(patULong dim,bioExpressionRepository* theRep) ;
  void setExpression(bioExpression* theExpr, 
		     patULong row, 
		     patULong col, 
		     patError*& err) ;
  bioExpression* getExpression(patULong row, 
			       patULong col, 
			       patError*& err) ;
  patULong getSize() const ;
 private:
  // The matrix contains the ID of the expressions
  vector<vector< patULong > > theMatrix ;
  bioExpressionRepository* theRepository ;
  
};
#endif
