//-*-c++-*------------------------------------------------------------
//
// File name : bioExprGreaterOrEqual.h
// @date   Thu Apr 19 07:24:00 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprGreaterOrEqual_h
#define bioExprGreaterOrEqual_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprGreaterOrEqual: public bioExpression {
 public:
  bioExprGreaterOrEqual(bioExpression* l, bioExpression* r) ;
  ~bioExprGreaterOrEqual() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
