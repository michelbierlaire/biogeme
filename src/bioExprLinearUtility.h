//-*-c++-*------------------------------------------------------------
//
// File name : bioExprLinearUtility.h
// @date   Wed Jul 10 08:07:11 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprLinearUtility_h
#define bioExprLinearUtility_h

#include <map>
#include "bioExpression.h"
#include "bioString.h"

class bioLinearTerm {
public:
  bioExpression* theBeta ;
  bioUInt theBetaId ;
  bioString theBetaName ;
  bioExpression* theVar ;
  bioUInt theVarId ;
  bioString theVarName ;
};

class bioExprLinearUtility: public bioExpression {
public:
  bioExprLinearUtility(std::vector<bioLinearTerm> t) ;
  ~bioExprLinearUtility() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  virtual bioString print(bioBoolean hp = false) const ;
protected:
  std::vector<bioLinearTerm > listOfTerms ;
  std::map<bioUInt,bioString> theFriend ;
};


#endif
