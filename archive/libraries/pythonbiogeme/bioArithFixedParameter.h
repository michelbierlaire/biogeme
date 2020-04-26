//-*-c++-*------------------------------------------------------------
//
// File name : bioArithFixedParameter.h
// Author :    Michel Bierlaire
// Date :      Fri Apr 22 08:14:32 2011
//
//--------------------------------------------------------------------

#ifndef bioArithFixedParameter_h
#define bioArithFixedParameter_h

#include "bioArithLiteral.h"
class bioFixedParameter ;

/*!
Class implementing the node for variables in an expression
*/

class bioArithFixedParameter: public bioArithLiteral {

public:
  bioArithFixedParameter(bioExpressionRepository* rep, 
			 patULong par,
			 patULong uniqueId,
			 patULong bId) ;
  
 public:
  
  virtual bioArithFixedParameter* getDeepCopy(bioExpressionRepository* rep,
					      patError*& err) const ;
  virtual bioArithFixedParameter* getShallowCopy(bioExpressionRepository* rep,
					      patError*& err) const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;


 protected:

  patULong theParameterId ;
  const bioFixedParameter* theParameter ;
};

#endif
