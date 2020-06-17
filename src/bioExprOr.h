//-*-c++-*------------------------------------------------------------
//
// File name : bioExprOr.h
// @date   Sun Apr 29 12:03:39 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprOr_h
#define bioExprOr_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprOr: public bioExpression {
 public:
  bioExprOr(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprOr() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
