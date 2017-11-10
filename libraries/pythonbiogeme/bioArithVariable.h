//-*-c++-*------------------------------------------------------------
//
// File name : bioArithVariable.h
// Author :    Michel Bierlaire
// Date :      Thu Apr 21 07:22:57 2011
//
//--------------------------------------------------------------------

#ifndef bioArithVariable_h
#define bioArithVariable_h

#include "bioArithLiteral.h"
class bioVariable ;

/*!
Class implementing the node for variables in an expression
*/

class bioArithVariable: public bioArithLiteral {

public:
  bioArithVariable(bioExpressionRepository* rep, patULong par, patULong uniqueId, patULong vId) ;
  
 public:
  
  virtual bioArithVariable* getDeepCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual bioArithVariable* getShallowCopy(bioExpressionRepository* rep,
					patError*& err) const ;
  virtual patReal getValue(patBoolean prepareGradient, patULong currentLap, patError*& err) ;

 protected:

  patULong theVariableId ;
  const bioVariable* theVariable ;
  patBoolean checkForMissingValues ;
  patReal missingValue ;


};

#endif
