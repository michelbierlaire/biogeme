 //-*-c++-*------------------------------------------------------------
//
// File name : bioExprFixedParameter.h
// @date   Thu Jul  4 19:15:59 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprFixedParameter_h
#define bioExprFixedParameter_h

#include "bioExprLiteral.h"
#include "bioString.h"

class bioExprFixedParameter: public bioExprLiteral {
 public:
  
  bioExprFixedParameter(bioUInt literalId, bioUInt parameterId, bioString name) ;
  ~bioExprFixedParameter() ;
  virtual bioString print(bioBoolean hp = false) const ;
  // Returns true is the expression contains at least one literal in
  // the list. Used to simplify the calculation of the derivatives
  virtual bioReal getLiteralValue() const ;

protected:
  bioUInt theParameterId ;
};


#endif
