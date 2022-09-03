//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNumeric.h
// @date   Fri Apr 13 15:11:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprNumeric_h
#define bioExprNumeric_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprNumeric: public bioExpression {
 public:
  bioExprNumeric(bioReal v) ;
  ~bioExprNumeric() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioReal value ;
};
#endif
