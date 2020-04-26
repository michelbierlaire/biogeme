 //-*-c++-*------------------------------------------------------------
//
// File name : bioExprFreeParameter.h
// @date   Thu Jul  4 18:55:15 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprFreeParameter_h
#define bioExprFreeParameter_h

#include "bioExprLiteral.h"
#include "bioString.h"

class bioExprFreeParameter: public bioExprLiteral {
 public:
  
  bioExprFreeParameter(bioUInt literalId, bioUInt parameterId, bioString name) ;
  ~bioExprFreeParameter() ;
  virtual bioString print(bioBoolean hp = false) const ;
  virtual bioReal getLiteralValue() const ;

protected:
  bioUInt theParameterId ;
};


#endif
