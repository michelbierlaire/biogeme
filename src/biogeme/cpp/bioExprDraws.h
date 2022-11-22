//-*-c++-*------------------------------------------------------------
//
// File name : bioExprDraws.h
// @date   Mon May  7 10:23:08 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------

#ifndef bioExprDraws_h
#define bioExprDraws_h

#include "bioExprLiteral.h"
#include "bioString.h"

class bioExprDraws: public bioExprLiteral {
 public:
  
  bioExprDraws(bioUInt uniqueId, bioUInt drawId, bioString name) ;
  ~bioExprDraws() ;
  virtual bioString print(bioBoolean hp = false) const ;
  virtual void setDrawIndex(bioUInt* d) ;
  virtual bioReal getLiteralValue() const ;
protected:
  bioUInt theDrawId ;
  bioUInt* drawIndex ;
};


#endif
