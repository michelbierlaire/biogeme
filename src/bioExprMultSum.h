//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMultSum.h
// @date   Wed Apr 18 11:04:12 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMultSum_h
#define bioExprMultSum_h

#include <vector>
#include "bioExpression.h"
#include "bioString.h"

class bioExprMultSum: public bioExpression {
 public:
  bioExprMultSum(std::vector<bioExpression*> e) ;
  ~bioExprMultSum() ;
  
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  std::vector<bioExpression*> expressions ;
};


#endif
