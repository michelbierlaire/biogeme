//-*-c++-*------------------------------------------------------------
//
// File name : bioExprElem.h
// @date   Wed Apr 18 10:31:22 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprElem_h
#define bioExprElem_h

#include <map>
#include "bioExpression.h"
#include "bioString.h"

class bioExprElem: public bioExpression {
 public:
  bioExprElem(bioExpression* k, std::map<bioUInt,bioExpression*> d) ;
  ~bioExprElem() ;
  
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  bioExpression* key ;
  std::map<bioUInt,bioExpression*> dictOfExpressions ;

};


#endif
