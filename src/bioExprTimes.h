//-*-c++-*------------------------------------------------------------
//
// File name : bioExprTimes.h
// @date   Fri Apr 13 11:39:11 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprTimes_h
#define bioExprTimes_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprTimes: public bioExpression {
 public:
  bioExprTimes(bioSmartPointer<bioExpression>  l, bioSmartPointer<bioExpression>  r) ;
  ~bioExprTimes() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;


  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  left ;
  bioSmartPointer<bioExpression>  right ;
};


#endif
