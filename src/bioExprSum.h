//-*-c++-*------------------------------------------------------------
//
// File name : bioExprSum.h
// @date   Fri Apr 13 14:56:25 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprSum_h
#define bioExprSum_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprSum: public bioExpression {
 public:
  bioExprSum(bioExpression* c, std::vector< std::vector<bioReal> >* d) ;
  ~bioExprSum() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* child ;
  std::vector< std::vector<bioReal> >* data ;
};
#endif
