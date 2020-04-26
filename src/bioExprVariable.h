 //-*-c++-*------------------------------------------------------------
//
// File name : bioExprVariable.h
// @date   Thu Jul  4 19:24:56 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprVariable_h
#define bioExprVariable_h

#include "bioExprLiteral.h"
#include "bioString.h"

class bioExprVariable: public bioExprLiteral {
 public:
  
  bioExprVariable(bioUInt literalId, bioUInt variableId, bioString name) ;
  ~bioExprVariable() ;
  virtual bioString print(bioBoolean hp = false) const ;
  // Returns true is the expression contains at least one literal in
  // the list. Used to simplify the calculation of the derivatives
  virtual bioReal getLiteralValue() const ;

protected:
  bioUInt theVariableId ;
};


#endif
