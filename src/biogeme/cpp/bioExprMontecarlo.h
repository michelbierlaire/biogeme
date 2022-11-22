//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMontecarlo.h
// @date   Tue May  8 10:30:53 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMontecarlo_h
#define bioExprMontecarlo_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMontecarlo: public bioExpression {
 public:
  bioExprMontecarlo(bioExpression* c) ;
  ~bioExprMontecarlo() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioUInt drawIndex ;
  bioExpression* child ;
};
#endif
