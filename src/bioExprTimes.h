//-*-c++-*------------------------------------------------------------
//
// File name : bioExprTimes.h
// @date   Fri Apr 13 11:39:11 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprTimes_h
#define bioExprTimes_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprTimes: public bioExpression {
 public:
  bioExprTimes(bioExpression* l, bioExpression* r) ;
  ~bioExprTimes() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;


  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* left ;
  bioExpression* right ;
};


#endif
