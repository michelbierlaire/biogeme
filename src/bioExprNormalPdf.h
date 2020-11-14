//-*-c++-*------------------------------------------------------------
//
// File name : bioExprNormalPdf.h
// @date   Tue Aug 20 08:21:35 2019
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioExprNormalPdf_h
#define bioExprNormalPdf_h

#include "bioExpression.h"
#include "bioString.h"
#include "bioNormalPdf.h"

class bioExprNormalPdf: public bioExpression {
 public:
  bioExprNormalPdf(bioExpression* c) ;
  ~bioExprNormalPdf() ;
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;
  
  virtual bioString print(bioBoolean hp = false) const ;
  
protected:
  bioExpression* child ;
};
#endif
