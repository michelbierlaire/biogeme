//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPower.h
// @date   Fri Apr 13 12:20:33 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprPower_h
#define bioExprPower_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprPower: public bioExpression {
 public:
  bioExprPower(bioExpression* l, bioExpression* r) ;
  ~bioExprPower() ;
  virtual bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};
#endif
