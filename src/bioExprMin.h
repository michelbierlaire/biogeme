//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMin.h
// @date   Mon Oct 15 15:34:34 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMin_h
#define bioExprMin_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMin: public bioExpression {
 public:
  bioExprMin(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprMin() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
