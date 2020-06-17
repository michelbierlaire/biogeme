//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLessOrEqual.h
// @date   Thu Apr 19 07:22:08 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLessOrEqual_h
#define bioExprLessOrEqual_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprLessOrEqual: public bioExpression {
 public:
  bioExprLessOrEqual(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprLessOrEqual() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
