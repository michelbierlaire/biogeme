//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLessOrEqual.h
// @date   Thu Apr 19 07:22:08 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLessOrEqual_h
#define bioExprLessOrEqual_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprLessOrEqual: public bioExpression {
 public:
  bioExprLessOrEqual(bioExpression* l, bioExpression* r) ;
  ~bioExprLessOrEqual() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
