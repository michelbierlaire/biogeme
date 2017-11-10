//-*-c++-*------------------------------------------------------------
//
// File name : bioArithCompositeLiteral.h
// Author :    Michel Bierlaire
// Date :      Thu Apr 21 09:09:18 2011
//
//--------------------------------------------------------------------

#ifndef bioArithCompositeLiteral_h
#define bioArithCompositeLiteral_h

#include "bioArithLiteral.h"
class bioCompositeLiteral ;

/*!
Class implementing the node for composite literals in an expression
*/

class bioArithCompositeLiteral: public bioArithLiteral {

public:
  
 public:
  
  virtual bioArithCompositeLiteral* getDeepCopy(bioExpressionRepository* rep,
						patError*& err) const ;
  virtual bioArithCompositeLiteral* getShallowCopy(bioExpressionRepository* rep,
						patError*& err) const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;

 protected:

  patULong theCompositeLiteralId ;

private: 
  bioArithCompositeLiteral(bioExpressionRepository* rep,
			   patULong par, 
			   patULong uniqueId, 
			   patULong vId) ;

};

#endif
