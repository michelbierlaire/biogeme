//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNotEqual.h
// @date   Thu Apr 19 07:19:23 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprNotEqual_h
#define bioExprNotEqual_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprNotEqual: public bioExpression {
 public:
  bioExprNotEqual(bioExpression* l, bioExpression* r) ;
  ~bioExprNotEqual() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
