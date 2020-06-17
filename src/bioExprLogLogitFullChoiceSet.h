//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLogLogitFullChoiceSet.h
// @date   Fri Jul  5 11:14:07 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprLogLogitFullChoiceSet_h
#define bioExprLogLogitFullChoiceSet_h

#include <map>
#include "bioExpression.h"
#include "bioString.h"

class bioExprLogLogitFullChoiceSet: public bioExpression {
 public:
  bioExprLogLogitFullChoiceSet(bioSmartPointer<bioExpression>  c, std::map<bioUInt,bioSmartPointer<bioExpression> > u) ;
  ~bioExprLogLogitFullChoiceSet() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioSmartPointer<bioExpression>  choice ;
  std::map<bioUInt,bioSmartPointer<bioExpression> > utilities ;
};


#endif
