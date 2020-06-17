//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLogLogit.h
// @date   Fri Apr 13 15:14:17 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLogLogit_h
#define bioExprLogLogit_h

#include <map>
#include "bioExpression.h"
#include "bioString.h"

class bioExprLogLogit: public bioExpression {
 public:
  bioExprLogLogit(bioSmartPointer<bioExpression>  c, std::map<bioUInt,bioSmartPointer<bioExpression> > u, std::map<bioUInt,bioSmartPointer<bioExpression> > a) ;
  ~bioExprLogLogit() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  choice ;
  std::map<bioUInt,bioSmartPointer<bioExpression> > utilities ;
  std::map<bioUInt,bioSmartPointer<bioExpression> > availabilities ;
};


#endif
