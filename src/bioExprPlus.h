//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPlus.h
// @date   Fri Apr 13 10:26:01 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprPlus_h
#define bioExprPlus_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprPlus: public bioExpression {
 public:
  bioExprPlus(bioExpression* l, bioExpression* r) ;
  ~bioExprPlus() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
 protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
