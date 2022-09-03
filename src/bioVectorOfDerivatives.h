//-*-c++-*------------------------------------------------------------
//
// File name : bioVectorOfDerivatives.h
// @date   Wed Oct 20 17:54:36 2021
// @author Michel Bierlaire
//
//--------------------------------------------------------------------

#ifndef bioVectorOfDerivatives_h
#define bioVectorOfDerivatives_h

#include <vector>
#include "bioDerivatives.h"

class bioVectorOfDerivatives {

 public:
  bioVectorOfDerivatives() ;
  void onlyOne(bioDerivatives d) ;
  void resizeAll(bioUInt n) ;
  void setEverythingToZero() ;
  void setDerivativesToZero() ;
  void dealWithNumericalIssues() ;
  void set_with_g(bioBoolean yes);
  void set_with_h(bioBoolean yes);
  void set_with_bhhh(bioBoolean yes);
  bioUInt getSize() const ;
  bioBoolean with_g() const ;
  bioBoolean with_h() const ;
  bioBoolean with_bhhh() const ;
  void aggregate(bioDerivatives d) ;
  void disaggregate(bioDerivatives d) ;
  void aggregate(bioVectorOfDerivatives d) ;
  void disaggregate(bioVectorOfDerivatives d) ;
  void clear() ;
 public:
  std::vector< bioDerivatives > theDerivatives ;
private:
  bioBoolean wg ;
  bioBoolean wh ;
  bioBoolean wbhhh ;
} ;
#endif
