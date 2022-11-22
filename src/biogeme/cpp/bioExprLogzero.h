//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprLogzero.h
// @date   Mon Oct 24 09:47:30 2022
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLogzero_h
#define bioExprLogzero_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprLogzero: public bioExpression {
 public:
  bioExprLogzero(bioExpression* c) ;
  ~bioExprLogzero() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						       bioBoolean gradient,
						       bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioExpression* child ;
};
#endif
