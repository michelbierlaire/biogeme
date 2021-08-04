//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDivide.h
// @date   Fri Apr 13 11:56:49 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprDivide_h
#define bioExprDivide_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprDivide: public bioExpression {
 public:
  bioExprDivide(bioExpression* l, bioExpression* r) ;
  ~bioExprDivide() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;


  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
