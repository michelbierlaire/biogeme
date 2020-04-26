//-*-c++-*------------------------------------------------------------
//
// File name : bioDerivatives.h
// @date   Fri Apr 13 10:29:32 2018
// @author Michel Bierlaire
// @version Revision 1.0
//
//--------------------------------------------------------------------


#ifndef bioDerivatives_h
#define bioDerivatives_h

#include <vector>
#include "bioTypes.h"

class bioDerivatives {
 public:
  bioDerivatives(bioUInt n) ;
  void setDerivativesToZero() ;
  void setGradientToZero() ;
  void setHessianToZero() ;
  bioUInt getSize() const ;
  bioReal f ;
  std::vector<bioReal> g ;
  std::vector<std::vector<bioReal> > h ;
};

std::ostream& operator<<(std::ostream &str, const bioDerivatives& x) ;

#endif
