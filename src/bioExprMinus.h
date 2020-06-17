//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMinus.h
// @date   Fri Apr 13 11:37:45 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMinus_h
#define bioExprMinus_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMinus: public bioExpression {
 public:
  bioExprMinus(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprMinus() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient, 
								 bioBoolean hessian) ;


  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
