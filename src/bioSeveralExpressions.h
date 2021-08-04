//-*-c++-*------------------------------------------------------------
//
// File name : bioSeveralExpressions.h
// @date   Wed Mar  3 18:03:24 2021
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioSeveralExpressions_h
#define bioSeveralExpressions_h

#include <vector>
#include "bioExpression.h"
#include "bioString.h"

class bioSeveralExpressions: public bioExpression {
public:
  bioSeveralExpressions(std::vector<bioExpression*> exprs) ;
  ~bioSeveralExpressions() ;
  virtual bioString print(bioBoolean hp = false) const ;
  virtual bioReal getValue() ;
  std::vector<bioReal > getValues() ;
  const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
					       bioBoolean gradient,
					       bioBoolean hessian) ;
  std::vector<const bioDerivatives* >
  getAllValueAndDerivatives(std::vector<bioUInt> literalIds,
			    bioBoolean gradient,
			    bioBoolean hessian) ;
private:
  std::vector<bioExpression*> theExpressions ;
};
#endif
