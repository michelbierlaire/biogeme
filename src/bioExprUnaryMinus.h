//-*-c++-*------------------------------------------------------------
//
// File name : bioExprUnaryMinus.h
// @date   Fri Apr 13 14:53:04 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprUnaryMinus_h
#define bioExprUnaryMinus_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprUnaryMinus: public bioExpression {
 public:
  bioExprUnaryMinus(bioSmartPointer<bioExpression> c) ;
  ~bioExprUnaryMinus() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioSmartPointer<bioExpression> child ;
};
#endif
