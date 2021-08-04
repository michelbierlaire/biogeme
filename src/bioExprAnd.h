//-*-c++-*------------------------------------------------------------
//
// File name : bioExprAnd.h
// @date   Sun Apr 29 11:58:20 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprAnd_h
#define bioExprAnd_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprAnd: public bioExpression {
 public:
  bioExprAnd(bioExpression* l, bioExpression* r) ;
  ~bioExprAnd() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
