//-*-c++-*------------------------------------------------------------
//
// File name : bioArithRandomVariable.h
// Author :    Michel Bierlaire
// Date :      Fri Apr 22 07:36:34 2011
//
//--------------------------------------------------------------------

#ifndef bioArithRandomVariable_h
#define bioArithRandomVariable_h

#include "bioArithLiteral.h"
class bioRandomVariable ;

/*!
Class implementing the node for a random variable used in an integral
*/

class bioArithRandomVariable: public bioArithLiteral {

public:
  bioArithRandomVariable(bioExpressionRepository* rep,
			 patULong par,
			 patULong uniqueId, 
			 patULong rvId) ;
  
 public:
  
  virtual bioArithRandomVariable* getDeepCopy(bioExpressionRepository* rep,
					      patError*& err) const ;
  virtual bioArithRandomVariable* getShallowCopy(bioExpressionRepository* rep,
					      patError*& err) const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;
 protected:

  patULong theRandomVariableId ;

};

#endif
