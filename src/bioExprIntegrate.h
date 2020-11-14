//-*-c++-*------------------------------------------------------------
//
// File name : bioExprIntegrate.h
// @date   Wed May  9 17:38:09 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprIntegrate_h
#define bioExprIntegrate_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprIntegrate: public bioExpression {
 public:
  bioExprIntegrate(bioExpression* c, bioUInt lid) ;
  ~bioExprIntegrate() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioExpression* child ;
  bioUInt rvId ;
};
#endif
