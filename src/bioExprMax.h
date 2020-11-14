//-*-c++-*------------------------------------------------------------
//
// File name : bioExprMax.h
// @date   Mon Oct 15 15:38:01 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprMax_h
#define bioExprMax_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprMax: public bioExpression {
 public:
  bioExprMax(bioExpression* l, bioExpression* r) ;
  ~bioExprMax() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
