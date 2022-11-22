//-*-c++-*------------------------------------------------------------
//
// File name : bioExprRandomVariable.h
// @date   Wed May  9 17:15:40 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprRandomVariable_h
#define bioExprRandomVariable_h

#include "bioExprLiteral.h"
#include "bioString.h"

class bioExprRandomVariable: public bioExprLiteral {
 public:
  
  bioExprRandomVariable(bioUInt literalId, bioUInt id, bioString name) ;
  ~bioExprRandomVariable() ;
  virtual bioString print(bioBoolean hp = false) const ;
  virtual void setRandomVariableValuePtr(bioUInt id, bioReal* v) ;
  virtual bioReal getLiteralValue() const ;
protected:
  bioUInt rvId ;
  bioReal* valuePtr ;
};


#endif
