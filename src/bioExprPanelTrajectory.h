//-*-c++-*------------------------------------------------------------
//
// File name : bioExprPanelTrajectory.h
// @date   Mon May 21 13:48:02 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprPanelTrajectory_h
#define bioExprPanelTrajectory_h

#include "bioExpression.h"
#include "bioString.h"

class bioExprPanelTrajectory: public bioExpression {
 public:
  bioExprPanelTrajectory(bioExpression* c) ;
  ~bioExprPanelTrajectory() ;
  bioExprPanelTrajectory(const bioExprPanelTrajectory&) = delete;
  void operator=(const bioExprPanelTrajectory&) = delete;  
  virtual const bioDerivatives* getValueAndDerivatives(std::vector<bioUInt> literalIds,
						 bioBoolean gradient,
						 bioBoolean hessian) ;

  virtual bioString print(bioBoolean hp = false) const ;
  virtual void setRowIndex(bioUInt* i) ;

 protected:
  bioUInt theRowIndex ;
  bioExpression* child ;

};
#endif
