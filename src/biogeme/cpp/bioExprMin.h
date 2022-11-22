//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMin.h
// @date   Mon Oct 15 15:34:34 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMin_h
#define bioExprMin_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMin: public bioExpression {
 public:
  bioExprMin(bioExpression* l, bioExpression* r) ;
  ~bioExprMin() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
