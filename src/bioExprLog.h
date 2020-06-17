//-*-c++-*----------------------%--------------------------------------
//
// File name : bioExprLog.h
// @date   Tue Apr 17 12:16:20 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprLog_h
#define bioExprLog_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprLog: public bioExpression {
 public:
  bioExprLog(bioSmartPointer<bioExpression>  c) ;
  ~bioExprLog() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioSmartPointer<bioExpression>  child ;
};
#endif
