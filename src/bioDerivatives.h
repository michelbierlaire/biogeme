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
  bioDerivatives() ;
  bioDerivatives& operator+=(const bioDerivatives& rhs) ;
  void clear() ;
  void resize(bioUInt n) ;
  void setEverythingToZero() ;
  void setDerivativesToZero() ;
  void computeBhhh() ;
  bioUInt getSize() const ;
  void dealWithNumericalIssues() ;
public:
  bioBoolean with_g ;
  bioBoolean with_h ;
  bioBoolean with_bhhh ;
  bioReal bhhh_weight ;
  bioReal f ;
  std::vector<bioReal> g ;
  std::vector<std::vector<bioReal> > h ;
  std::vector<std::vector<bioReal> > bhhh ;
};

std::ostream& operator<<(std::ostream &str, const bioDerivatives& x) ;


#endif
