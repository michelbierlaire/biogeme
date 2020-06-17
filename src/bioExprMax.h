//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMax.h
// @date   Mon Oct 15 15:38:01 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMax_h
#define bioExprMax_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMax: public bioExpression {
 public:
  bioExprMax(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprMax() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
