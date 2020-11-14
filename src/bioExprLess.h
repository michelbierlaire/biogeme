//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLess.h
// @date   Thu Apr 19 07:25:51 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLess_h
#define bioExprLess_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprLess: public bioExpression {
 public:
  bioExprLess(bioExpression* l, bioExpression* r) ;
  ~bioExprLess() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
