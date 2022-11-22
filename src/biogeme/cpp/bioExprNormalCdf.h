//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNormalCdf.h
// @date   Wed May 30 16:17:29 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprNormalCdf_h
#define bioExprNormalCdf_h

#include "bioExpression.h"
#include "bioString.h"
#include "bioNormalCdf.h"

class bioExprNormalCdf: public bioExpression {
 public:
  bioExprNormalCdf(bioExpression* c) ;
  ~bioExprNormalCdf() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
  
protected:
  bioNormalCdf theNormalCdf ;
  bioExpression* child ;
};
#endif
