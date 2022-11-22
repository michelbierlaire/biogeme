//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprExp.h
// @date   Tue Apr 17 12:12:43 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprExp_h
#define bioExprExp_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprExp: public bioExpression {
 public:
  bioExprExp(bioExpression* c) ;
  ~bioExprExp() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioExpression* child ;
};
#endif
