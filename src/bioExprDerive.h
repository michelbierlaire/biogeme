//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDerive.h
// @date   Tue May  1 21:02:47 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprDerive_h
#define bioExprDerive_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprDerive: public bioExpression {
 public:
  bioExprDerive(bioSmartPointer<bioExpression>  c, bioUInt lid) ;
  ~bioExprDerive() ;
  virtual bioSmartPointer<bioDerivatives> getValueAndDerivatives(std::vector<bioUInt> literalIds,
								 bioBoolean gradient,
								 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;

 protected:
  bioSmartPointer<bioExpression>  child ;
  bioUInt literalId ;
};
#endif
