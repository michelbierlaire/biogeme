//-*-c++-*------------------------------------------------------------
//
// File name : bioExprEqual.h
// @date   Thu Apr 19 07:15:57 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprEqual_h
#define bioExprEqual_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprEqual: public bioExpression {
 public:
  bioExprEqual(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprEqual() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
